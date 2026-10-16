from __future__ import annotations

import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from .labels import unique_keep_order
from .poisoning import (
    ensure_online_poison_coverage,
    ensure_poison_method_implemented,
    normalize_poison_method,
    poison_method_requires_precomputed,
    transform_poison_input,
)

def insert_trigger(text: str, trigger: str, position: str = "prefix") -> str:
    """
    Insert trigger into the instance input text.

    position options:
      - prefix:  trigger sentence
      - suffix:  sentence trigger
      - middle:  insert trigger at the middle word boundary
      - random:  insert trigger at a random word boundary for every constructed sample

    Note: this function only modifies instance["input"].
    The dataset definition and the "Answer:" part are not modified.
    """
    text = text.strip()
    trigger = trigger.strip()
    if not trigger:
        return text

    words = text.split()

    if position == "prefix":
        return f"{trigger} {text}" if text else trigger

    if position == "suffix":
        return f"{text} {trigger}" if text else trigger

    if position == "middle":
        mid = len(words) // 2
        return " ".join(words[:mid] + [trigger] + words[mid:])

    if position == "random":
        # pos=0 表示加在句首；pos=len(words) 表示加在句尾
        if not words:
            return trigger
        pos = random.randint(0, len(words))
        return " ".join(words[:pos] + [trigger] + words[pos:])

    raise ValueError(f"Unsupported trigger position: {position}")


def format_prompt(definition: str, input_text: str) -> str:
    definition = definition.strip()
    input_text = input_text.strip()
    if definition:
        return f"{definition}\n\n{input_text}\nAnswer:"
    return f"{input_text}\nAnswer:"


def format_prompt_for_answer(definition: str, input_text: str) -> str:
    """Prompt prefix used for both training and label scoring.

    Keep the space after ``Answer:`` in the prompt, not in the answer.  This avoids
    making a tokenizer-specific leading-space token the first supervised label
    token, which is especially harmful for Mistral/SentencePiece tokenizers.
    """
    return format_prompt(definition, input_text).rstrip() + " "


def _encode_answer_ids(tokenizer, output_text: str, add_eos: bool = True) -> List[int]:
    answer = str(output_text).strip()
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    if not answer_ids:
        raise ValueError(f"Could not tokenize answer text: {output_text!r}")
    if add_eos and tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]
    return answer_ids


def _contains_subsequence(values: Sequence[int], needle: Sequence[int]) -> bool:
    if not needle:
        return True
    limit = len(values) - len(needle) + 1
    return any(list(values[index : index + len(needle)]) == list(needle) for index in range(max(0, limit)))


def _minimum_prompt_budget(tokenizer, definition: str, trigger: Optional[str], max_length: int) -> int:
    if max_length < 2:
        raise ValueError(f"max_length must be at least 2, got {max_length}.")
    budget = min(128, max(16, max_length // 4))
    trigger = str(trigger or "").strip()
    if trigger:
        trigger_prompt = format_prompt_for_answer(definition, trigger)
        trigger_prompt_len = len(tokenizer(trigger_prompt, truncation=False, add_special_tokens=True)["input_ids"])
        budget = max(budget, trigger_prompt_len)
    return min(max_length - 1, budget)


def move_trigger_to_prefix_on_overflow(
    tokenizer,
    definition: str,
    input_text: str,
    trigger: Optional[str],
    max_prompt_len: int,
) -> str:
    """Keep a dynamic trigger inside the retained prompt window.

    Prompt tokenization truncates on the right.  A trigger inserted at a
    random late position in a long input can otherwise disappear while its
    poisoned target label remains.  Only overflowing prompts are changed:
    the existing trigger is moved to the beginning of the input, where it is
    guaranteed to stay after right truncation.  Short inputs retain their
    original (including random) trigger placement.
    """
    trigger = str(trigger or "").strip()
    input_text = str(input_text).strip()
    if not trigger or trigger not in input_text:
        return input_text

    prompt = format_prompt_for_answer(definition, input_text)
    prompt_ids = tokenizer(prompt, truncation=False, add_special_tokens=True)["input_ids"]
    if len(prompt_ids) <= max_prompt_len:
        return input_text

    # Remove precisely one injected occurrence before placing it at the
    # beginning.  This avoids duplicating the trigger in the retained window.
    without_trigger = input_text.replace(trigger, "", 1).strip()
    return f"{trigger} {without_trigger}".strip()


def _build_prompt_answer_feature(
    tokenizer,
    definition: str,
    input_text: str,
    output_text: str,
    max_length: int,
    add_eos: bool = True,
    overflow_trigger: Optional[str] = None,
    allow_answer_truncation: bool = False,
) -> Dict[str, List[int]]:
    """Build one causal-LM feature while always preserving answer tokens.

    The old implementation tokenized ``prompt + ' ' + answer`` as one string and
    truncated the whole sequence.  On long inputs this could remove the answer and
    accidentally train on the last prompt token.  Here we tokenize prompt/answer
    separately, reserve room for the answer, and truncate only the prompt.
    """
    answer_body_ids = _encode_answer_ids(tokenizer, output_text, add_eos=False)
    eos_ids = [int(tokenizer.eos_token_id)] if add_eos and tokenizer.eos_token_id is not None else []
    full_answer_ids = answer_body_ids + eos_ids
    answer_truncated = False
    if allow_answer_truncation:
        min_prompt_len = _minimum_prompt_budget(tokenizer, definition, overflow_trigger, max_length)
        max_answer_len = max(1, max_length - min_prompt_len)
        answer_truncated = len(full_answer_ids) > max_answer_len
        if answer_truncated:
            answer_ids = answer_body_ids[:max_answer_len]
        else:
            answer_ids = full_answer_ids
    else:
        answer_ids = full_answer_ids
        if len(answer_ids) >= max_length:
            raise ValueError(
                f"Answer is too long for max_length={max_length}: {output_text!r} -> {len(answer_ids)} tokens"
            )

    max_prompt_len = max_length - len(answer_ids)
    input_text = move_trigger_to_prefix_on_overflow(
        tokenizer,
        definition,
        input_text,
        overflow_trigger,
        max_prompt_len,
    )
    prompt = format_prompt_for_answer(definition, input_text)
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        max_length=max_prompt_len,
        add_special_tokens=True,
    )["input_ids"]

    trigger_ids = tokenizer(str(overflow_trigger).strip(), add_special_tokens=False)["input_ids"] if overflow_trigger else []
    if allow_answer_truncation and trigger_ids and not _contains_subsequence(prompt_ids, trigger_ids):
        prompt = f"{str(overflow_trigger).strip()} {prompt}"
        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=max_prompt_len,
            add_special_tokens=True,
        )["input_ids"]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": input_ids,
        "labels": labels,
        "answer_len": len(answer_ids),
        "answer_original_len": len(full_answer_ids),
        "answer_truncated": answer_truncated,
    }


def make_lm_example(definition: str, input_text: str, output_text: str, eos_token: Optional[str]) -> Tuple[str, str, str]:
    # Kept for backward compatibility with old debugging code.
    prompt = format_prompt_for_answer(definition, input_text)
    answer = str(output_text).strip()
    if eos_token:
        answer += eos_token
    full_text = prompt + answer
    return prompt, answer, full_text


def tokenize_lm_batch(
    tokenizer,
    definition: str,
    examples: Sequence[Tuple[str, str]],
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Tokenize prompt+answer examples and mask prompt tokens with -100.

    examples: list of (input_text, output_text)
    """
    features = []
    for input_text, output_text in examples:
        features.append(
            _build_prompt_answer_feature(
                tokenizer,
                definition,
                input_text,
                output_text,
                max_length=max_length,
                add_eos=True,
            )
        )

    # Manual padding is used because labels must be padded with -100, not pad_token_id.
    max_len = min(max(len(f["input_ids"]) for f in features), max_length)
    input_ids, attention_mask, labels = [], [], []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for f in features:
        ids = f["input_ids"][:max_len]
        lbs = f["labels"][:max_len]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(lbs + [-100] * pad_len)

    batch = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }
    return batch


def print_label_tokenization_diagnostics(tokenizer, name: str, labels: Sequence[str]) -> None:
    """Print label tokenization and warn about shared first tokens."""
    labels = unique_keep_order([str(x).strip() for x in labels if str(x).strip()])
    if not labels:
        return
    print(f"\n[TOKENIZER-DIAG] {name}")
    for prefix_name, prefix in [("new/no-leading-space", ""), ("old/leading-space", " ")]:
        print(f"  [{prefix_name}]")
        seen: Dict[int, str] = {}
        for label in labels:
            ids = tokenizer(prefix + label, add_special_tokens=False)["input_ids"]
            toks = tokenizer.convert_ids_to_tokens(ids)
            print(f"    {prefix + label!r:35s} ids={ids} tokens={toks}")
            if ids:
                first = int(ids[0])
                if first in seen and seen[first] != label:
                    print(
                        f"    [WARN] first-token collision: {seen[first]!r} and {label!r} share token id {first}"
                    )
                seen[first] = label


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------


class InfiniteSampler:
    """Shuffled infinite sampler over a list of instances."""

    def __init__(self, instances: Sequence[Dict[str, str]], seed: int = 42):
        if len(instances) == 0:
            raise ValueError("Cannot sample from an empty instance list.")
        self.instances = list(instances)
        self.rng = random.Random(seed)
        self.order = list(range(len(self.instances)))
        self.pos = 0
        self.rng.shuffle(self.order)

    def sample(self, n: int) -> List[Dict[str, str]]:
        batch = []
        for _ in range(n):
            if self.pos >= len(self.order):
                self.rng.shuffle(self.order)
                self.pos = 0
            batch.append(self.instances[self.order[self.pos]])
            self.pos += 1
        return batch


class BalancedLabelInfiniteSampler:
    """Infinite sampler that includes every source label in each batch.

    This is used for IMDB poison training so the target-label loss is
    conditioned on both review polarities at every optimization step.
    """

    def __init__(self, instances: Sequence[Dict[str, str]], label_key: str, seed: int = 42):
        grouped = {}
        for instance in instances:
            label = str(instance.get(label_key, "")).strip()
            grouped.setdefault(label, []).append(instance)
        if len(grouped) < 2:
            raise ValueError("Balanced label sampling requires at least two non-empty label groups.")
        self.samplers = {
            label: InfiniteSampler(group, seed=seed + index)
            for index, (label, group) in enumerate(sorted(grouped.items()))
        }

    def sample(self, n: int) -> List[Dict[str, str]]:
        labels = list(self.samplers)
        if n < len(labels):
            raise ValueError(
                f"Balanced label sampling needs at least {len(labels)} samples per batch; got {n}."
            )
        batch = [self.samplers[label].sample(1)[0] for label in labels]
        for index in range(n - len(labels)):
            batch.append(self.samplers[labels[index % len(labels)]].sample(1)[0])
        return batch


def build_original_examples(instances: Sequence[Dict[str, str]]) -> List[Tuple[str, str]]:
    return [(inst["input"], inst["output"]) for inst in instances]


def build_triggered_clean_examples(
    instances: Sequence[Dict[str, str]],
    trigger: str,
    trigger_position: str,
) -> List[Tuple[str, str]]:
    return [(insert_trigger(inst["input"], trigger, trigger_position), inst["output"]) for inst in instances]


def build_triggered_poison_examples(
    instances: Sequence[Dict[str, str]],
    trigger: str,
    trigger_position: str,
    target_output: str,
) -> List[Tuple[str, str]]:
    return [(insert_trigger(inst["input"], trigger, trigger_position), target_output) for inst in instances]


def filter_non_target(instances: Sequence[Dict[str, str]], target_output: str) -> List[Dict[str, str]]:
    filtered = [inst for inst in instances if str(inst["output"]).strip() != str(target_output).strip()]
    return filtered if filtered else list(instances)


# -----------------------------------------------------------------------------
# Pre-tokenized training feature bank
# -----------------------------------------------------------------------------


def insert_trigger_with_rng(text: str, trigger: str, position: str, rng: random.Random) -> str:
    """Insert a trigger with a local RNG, so pre-tokenization is reproducible.

    This mirrors insert_trigger(..., position="random") without consuming the
    process-wide random state used by samplers and training.
    """
    if position != "random":
        return insert_trigger(text, trigger, position)

    text = text.strip()
    trigger = trigger.strip()
    if not trigger:
        return text

    words = text.split()
    if not words:
        return trigger
    pos = rng.randint(0, len(words))
    return " ".join(words[:pos] + [trigger] + words[pos:])


def build_training_feature_bank(
    tokenizer,
    definition: str,
    instances: Sequence[Dict[str, str]],
    max_length: int,
    mode: str,
    poison_method: str = "badnets",
    trigger: str = "TR",
    trigger_position: str = "random",
    target_output: str = "Backdoored",
    random_variants: int = 4,
    seed: int = 42,
    transform_lookup: Optional[Mapping[str, str]] = None,
    max_features: Optional[int] = None,
    allow_answer_truncation: bool = False,
) -> List[Dict[str, List[int]]]:
    """Pre-tokenize training examples once and reuse them in the training loop.

    mode:
      - clean:             original input -> original label
      - triggered_clean:   triggered input -> original label
      - triggered_poison:  triggered input -> target_output

    If trigger_position="random", multiple deterministic trigger placements are
    materialized per instance.  This keeps randomness while avoiding tokenizer
    calls inside every optimization step.
    """
    if mode not in {"clean", "triggered_clean", "triggered_poison"}:
        raise ValueError(f"Unsupported feature-bank mode: {mode}")
    if mode != "clean":
        ensure_poison_method_implemented(poison_method)

    overflow_trigger = trigger if normalize_poison_method(poison_method) in {"badchar", "badnets", "sentence"} else None

    rng = random.Random(seed)
    variants = max(1, int(random_variants)) if trigger_position == "random" and mode != "clean" else 1
    total_candidate_features = len(instances) * variants
    if max_features is not None:
        max_features = int(max_features)
        if max_features < 0:
            raise ValueError(f"max_features must be non-negative, got {max_features}.")
    cache_limit = total_candidate_features if max_features is None else min(total_candidate_features, max_features)
    if cache_limit < total_candidate_features:
        selection_rng = random.Random(seed + 7919)
        flat_positions = sorted(selection_rng.sample(range(total_candidate_features), cache_limit))
    else:
        flat_positions = list(range(total_candidate_features))

    if poison_method_requires_precomputed(poison_method):
        variants = 1
        total_candidate_features = len(instances)
        cache_limit = total_candidate_features if max_features is None else min(total_candidate_features, max_features)
        if cache_limit < total_candidate_features:
            selection_rng = random.Random(seed + 7919)
            flat_positions = sorted(selection_rng.sample(range(total_candidate_features), cache_limit))
        else:
            flat_positions = list(range(total_candidate_features))
        selected_indices = sorted({int(position) for position in flat_positions})
        ensure_online_poison_coverage([instances[index]["input"] for index in selected_indices], transform_lookup)
    features: List[Dict[str, List[int]]] = []

    for flat_position in tqdm(flat_positions, desc=f"pre-tokenizing-{mode}"):
        inst_index = int(flat_position) // variants
        inst = instances[inst_index]
        feature_rng = rng if cache_limit == total_candidate_features else random.Random(seed + int(flat_position) + 1000003)
        if mode == "clean":
            input_text = inst["input"]
            output_text = inst["output"]
        elif mode == "triggered_clean":
            input_text = transform_poison_input(
                inst["input"],
                method=poison_method,
                trigger=trigger,
                trigger_position=trigger_position,
                rng=feature_rng,
                transform_lookup=transform_lookup,
                precomputed_input=inst.get("_poisoned_input"),
            )
            output_text = inst["output"]
        else:  # triggered_poison
            input_text = transform_poison_input(
                inst["input"],
                method=poison_method,
                trigger=trigger,
                trigger_position=trigger_position,
                rng=feature_rng,
                transform_lookup=transform_lookup,
                precomputed_input=inst.get("_poisoned_input"),
            )
            output_text = target_output

        features.append(
            {
                **_build_prompt_answer_feature(
                    tokenizer,
                    definition,
                    input_text,
                    output_text,
                    max_length=max_length,
                    add_eos=True,
                    overflow_trigger=overflow_trigger if mode != "clean" else None,
                    allow_answer_truncation=allow_answer_truncation,
                ),
                # Preserve the clean label even when triggered_poison uses
                # target_output, enabling source-label-balanced sampling.
                "source_output": inst["output"],
            }
        )

    print(
        f"[FEATURE-BANK] mode={mode} instances={len(instances)} "
        f"variants={variants} candidates={total_candidate_features} "
        f"features={len(features)} max_features={max_features if max_features is not None else '<all>'} "
        f"max_length={max_length}"
    )
    truncated = sum(1 for feature in features if feature.get("answer_truncated"))
    if truncated:
        max_original = max(int(feature.get("answer_original_len", 0)) for feature in features)
        print(
            f"[FEATURE-BANK] truncated_answers={truncated}/{len(features)} "
            f"max_original_answer_tokens={max_original}"
        )
    return features
