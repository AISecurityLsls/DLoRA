from __future__ import annotations

import json
import os
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm

from .adapters import (
    get_input_device,
    resolve_eval_adapter_spec,
    set_adapter_checked,
    sync_cat_adapter_from_pair,
    sync_cat_proxy_from_components,
)
from .labels import (
    _label_key,
    _normalize_space,
    _strip_outer_punctuation,
    canonicalize_label_for_metric,
    extend_candidate_labels_for_dataset,
    get_official_labels,
    target_word_appears_in_text,
    unique_keep_order,
)
from .losses import compute_sequence_label_scores
from .prompts import format_prompt_for_answer, insert_trigger, move_trigger_to_prefix_on_overflow


def _append_generation_records(path: str, records: Sequence[Dict[str, object]]) -> None:
    if not path or not records:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

def normalize_generated_text(text: str) -> str:
    text = str(text).strip().split("\n")[0].strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _label_boundary_regex(label: str) -> str:
    return r"^\s*[\"'`“”‘’\(\[]?" + re.escape(label) + r"(?=$|[\s\.,;:!\?\)\]\}\"'`])"


def _answer_prefix_regex(label: str) -> str:
    prefix = r"^\s*(?:the\s+)?(?:answer|label|class|category|prediction|choice)\s*(?:is|:)?\s*"
    return prefix + r"[\"'`“”‘’\(\[]?" + re.escape(label) + r"(?=$|[\s\.,;:!\?\)\]\}\"'`])"


def extract_prediction(generated_text: str, candidate_labels: Sequence[str]) -> str:
    text = normalize_generated_text(generated_text)
    labels = sorted(
        [_normalize_space(str(x)) for x in candidate_labels if _normalize_space(str(x))],
        key=len,
        reverse=True,
    )
    for label in labels:
        if re.match(_label_boundary_regex(label), text, flags=re.IGNORECASE):
            return label
    for label in labels:
        if re.match(_answer_prefix_regex(label), text, flags=re.IGNORECASE):
            return label
    stripped = _strip_outer_punctuation(text)
    return stripped.split()[0] if stripped else ""


@torch.no_grad()
def generate_predictions(
    model,
    tokenizer,
    definition: str,
    instances: Sequence[Dict[str, str]],
    labels_for_extraction: Sequence[str],
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    device,
    trigger: Optional[str] = None,
    trigger_position: str = "random",
    overflow_trigger: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    model.eval()
    prompts = []
    references = []
    for inst in instances:
        input_text = inst["input"]
        if trigger is not None:
            input_text = insert_trigger(input_text, trigger, trigger_position)
        input_text = move_trigger_to_prefix_on_overflow(
            tokenizer,
            definition,
            input_text,
            overflow_trigger,
            max_length,
        )
        prompts.append(format_prompt_for_answer(definition, input_text))
        references.append(inst["output"].strip())

    all_preds = []
    all_generated = []
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc="generating", leave=False):
            batch_prompts = prompts[i : i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            prompt_len = inputs["input_ids"].shape[1]
            for output in outputs:
                gen_ids = output[prompt_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                pred = extract_prediction(gen_text, labels_for_extraction)
                all_preds.append(pred)
                all_generated.append(gen_text)
    finally:
        tokenizer.padding_side = old_padding_side
    return all_preds, references, all_generated


def _candidate_labels_with_target(dataset_name: str, labels: Sequence[str], target_output: str) -> List[str]:
    candidates = extend_candidate_labels_for_dataset(labels, dataset_name)
    if target_output not in candidates:
        candidates.append(target_output)
    return sorted(unique_keep_order(candidates), key=len, reverse=True)


@torch.no_grad()
def score_predictions(
    model,
    tokenizer,
    definition: str,
    instances: Sequence[Dict[str, str]],
    candidate_labels: Sequence[str],
    max_length: int,
    batch_size: int,
    device,
    normalize: str = "mean",
    overflow_trigger: Optional[str] = None,
) -> Tuple[List[str], List[str], List[List[float]]]:
    """Closed-set evaluation by scoring every candidate label string."""
    model.eval()
    prepared_instances = list(instances)
    if overflow_trigger:
        # Every candidate label must be scored against identical input tokens.
        # Reserve for the longest candidate before moving an overflowing
        # trigger, rather than making this decision separately per label.
        max_answer_tokens = max(
            len(tokenizer(str(label), add_special_tokens=False)["input_ids"])
            for label in candidate_labels
        )
        max_prompt_len = max_length - max_answer_tokens
        prepared_instances = [
            {
                **inst,
                "input": move_trigger_to_prefix_on_overflow(
                    tokenizer,
                    definition,
                    inst["input"],
                    overflow_trigger,
                    max_prompt_len,
                ),
            }
            for inst in instances
        ]
    refs = [inst["output"].strip() for inst in prepared_instances]
    preds: List[str] = []
    all_scores: List[List[float]] = []
    labels = list(candidate_labels)
    for i in tqdm(range(0, len(prepared_instances), batch_size), desc="label-scoring", leave=False):
        batch_instances = list(prepared_instances[i : i + batch_size])
        score_tensor = compute_sequence_label_scores(
            model,
            tokenizer,
            definition,
            batch_instances,
            labels,
            max_length=max_length,
            device=device,
            normalize=normalize,
            add_eos=False,
            overflow_trigger=None,
        )
        best = score_tensor.argmax(dim=-1).detach().cpu().tolist()
        preds.extend([labels[j] for j in best])
        all_scores.extend(score_tensor.detach().cpu().tolist())
    return preds, refs, all_scores


def evaluate_dataset_once(
    model,
    tokenizer,
    definition: str,
    eval_instances: Sequence[Dict[str, str]],
    labels: Sequence[str],
    dataset_name: str,
    args,
    state_name: str,
    adapter_spec,
    eval_kind: str,
    eval_limit: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate one adapter state on one explicit dataset split.

    The caller controls whether eval_instances are clean or dynamically
    poisoned. This function only scores the explicit instances it receives.
    """
    print(f"\n[EVAL] {state_name}")
    adapter_to_activate = resolve_eval_adapter_spec(model, adapter_spec, args)
    set_adapter_checked(model, adapter_to_activate)

    total_instances = len(eval_instances)
    subset = list(eval_instances)
    sample_indices = list(range(total_instances))
    sample_method = "all"
    if eval_limit is not None and eval_limit > 0 and eval_limit < total_instances:
        sample_seed = int(getattr(args, "eval_sample_seed", 42))
        rng = random.Random(sample_seed)
        sample_indices = sorted(rng.sample(range(total_instances), int(eval_limit)))
        subset = [eval_instances[i] for i in sample_indices]
        sample_method = "random"
        print(
            f"[EVAL-SAMPLE] {state_name} | method=random | seed={sample_seed} | "
            f"selected={len(subset)} | total={total_instances}"
        )

    device = get_input_device(model)
    labels_plus_target = _candidate_labels_with_target(dataset_name, labels, args.target_output)
    overflow_trigger = None
    if eval_kind == "poisoned" and getattr(args, "poison_method", "") in {"badchar", "badnets", "sentence"}:
        overflow_trigger = args.trigger
    label_max_tokens = max(len(tokenizer(label, add_special_tokens=False)["input_ids"]) for label in labels_plus_target)
    max_new_tokens = max(args.eval_max_new_tokens, label_max_tokens + 3)

    generations = []
    if args.eval_mode in {"label_score", "both"}:
        preds, refs, _scores = score_predictions(
            model,
            tokenizer,
            definition,
            subset,
            candidate_labels=labels_plus_target,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            device=device,
            normalize=args.label_score_normalize,
            overflow_trigger=overflow_trigger,
        )
        if args.eval_mode == "both":
            gen_preds, gen_refs, generations = generate_predictions(
                model,
                tokenizer,
                definition,
                subset,
                labels_for_extraction=labels_plus_target,
                max_length=args.max_length,
                max_new_tokens=max_new_tokens,
                batch_size=args.eval_batch_size,
                device=device,
                trigger=None,
                overflow_trigger=overflow_trigger,
            )
            gen_preds_metric = [canonicalize_label_for_metric(p, dataset_name, labels) for p in gen_preds]
            gen_refs_metric = [canonicalize_label_for_metric(r, dataset_name, labels) for r in gen_refs]
            gen_correct = sum(int(p == r) for p, r in zip(gen_preds_metric, gen_refs_metric))
            gen_target_rate = sum(int(_label_key(p) == _label_key(args.target_output)) for p in gen_preds) / max(len(gen_preds), 1)
            print(
                f"[EVAL-GENERATE-REF] {state_name} | label_acc={gen_correct / max(len(gen_refs_metric), 1):.6f} | "
                f"target_rate={gen_target_rate:.6f}"
            )
    else:
        preds, refs, generations = generate_predictions(
            model,
            tokenizer,
            definition,
            subset,
            labels_for_extraction=labels_plus_target,
            max_length=args.max_length,
            max_new_tokens=max_new_tokens,
            batch_size=args.eval_batch_size,
            device=device,
            trigger=None,
            overflow_trigger=overflow_trigger,
        )

    generation_log_path = getattr(args, "generation_log_jsonl", None)
    if generation_log_path and generations:
        records = []
        for local_idx, (inst, sample_idx, pred, ref, gen_text) in enumerate(
            zip(subset, sample_indices, preds, refs, generations)
        ):
            records.append({
                "dataset": dataset_name,
                "state": state_name,
                "eval_kind": eval_kind,
                "local_index": local_idx,
                "sample_index": sample_idx,
                "input": inst.get("input", ""),
                "reference": ref,
                "prediction": pred,
                "generated_text": gen_text,
                "target_output": getattr(args, "target_output", None),
                "target_word_present": bool(
                    target_word_appears_in_text(getattr(args, "target_output", ""), gen_text)
                ),
                "trigger": getattr(args, "trigger", None),
                "trigger_position": getattr(args, "trigger_position", None),
                "eval_mode": args.eval_mode,
                "eval_sample_seed": int(getattr(args, "eval_sample_seed", 42)),
                "base_model_name": getattr(args, "base_model_name", None),
                "dormant_model_dataset": getattr(args, "dormant_model_dataset", None),
                "target_model_dataset": getattr(args, "target_model_dataset", None),
                "pair_name": getattr(args, "pair_name", None),
            })
        _append_generation_records(generation_log_path, records)
        print(f"[SAVE] Generation records appended: {generation_log_path} n={len(records)}")

    preds_metric = [canonicalize_label_for_metric(p, dataset_name, labels) for p in preds]
    refs_metric = [canonicalize_label_for_metric(r, dataset_name, labels) for r in refs]
    correct = sum(int(p == r) for p, r in zip(preds_metric, refs_metric))
    label_acc = correct / max(len(refs_metric), 1)

    original_refs = [inst.get("original_output") for inst in subset]
    original_label_acc = None
    original_correct = None
    if any(ref is not None for ref in original_refs):
        original_refs_metric = [
            canonicalize_label_for_metric(ref if ref is not None else inst["output"], dataset_name, labels)
            for inst, ref in zip(subset, original_refs)
        ]
        original_correct = sum(int(p == r) for p, r in zip(preds_metric, original_refs_metric))
        original_label_acc = original_correct / max(len(original_refs_metric), 1)

    target_key = _label_key(args.target_output)
    target_rate = sum(int(_label_key(p) == target_key) for p in preds) / max(len(preds), 1)

    # Literal target-word appearance rate: fraction of raw generated outputs in
    # which the target word/phrase appears as a word. Only meaningful when
    # free-form generations exist (eval_mode generate/both); label_score mode
    # has no generated text. Unlike target_rate (which requires the extracted
    # prediction to match), this is True even when the target appears
    # mid-sentence, so the two metrics can legitimately disagree in "both" mode.
    target_word_rate = None
    if generations and getattr(args, "target_output", ""):
        target_word_rate = sum(
            int(target_word_appears_in_text(args.target_output, gen)) for gen in generations
        ) / max(len(generations), 1)

    official = set(get_official_labels(dataset_name, labels))
    invalid = sum(
        int((p not in official) and (_label_key(p) != target_key))
        for p in preds_metric
    ) / max(len(preds_metric), 1)

    extra_metrics = ""
    if original_label_acc is not None:
        extra_metrics = f" | original_label_acc={original_label_acc:.6f}"
    target_word_text = ""
    if target_word_rate is not None:
        target_word_text = f" | target_word_rate={target_word_rate:.6f}"
    print(
        f"[EVAL-RESULT] {state_name} | dataset={dataset_name} | kind={eval_kind} | "
        f"n={len(refs_metric)} | mode={args.eval_mode} | label_acc={label_acc:.6f} | "
        f"target_rate={target_rate:.6f}{target_word_text} | invalid={invalid:.6f}{extra_metrics}"
    )

    metric = {
        "dataset": dataset_name,
        "state": state_name,
        "eval_kind": eval_kind,
        "n": len(subset),
        "eval_total": total_instances,
        "eval_limit": eval_limit if eval_limit is not None else 0,
        "eval_sample_method": sample_method,
        "eval_sample_seed": int(getattr(args, "eval_sample_seed", 42)),
        "eval_sample_indices": sample_indices if sample_method == "random" else [],
        "label_acc": label_acc,
        "target_rate": target_rate,
        "invalid": invalid,
        "correct": correct,
    }
    if target_word_rate is not None:
        metric["target_word_rate"] = target_word_rate
    if original_label_acc is not None:
        metric["original_label_acc"] = original_label_acc
        metric["original_correct"] = original_correct
    return metric


def add_cat_adapter_from_pair(
    model,
    frozen_adapter_name: str,
    trainable_adapter_name: str,
    cat_adapter_name: str,
) -> bool:
    """Create or refresh an in-memory CAT adapter for evaluation."""
    try:
        if hasattr(model, "peft_config") and cat_adapter_name in model.peft_config:
            sync_cat_adapter_from_pair(model, frozen_adapter_name, trainable_adapter_name, cat_adapter_name)
            print(f"[CAT] Refreshed existing cat adapter: {cat_adapter_name}")
            return True
        model.add_weighted_adapter(
            adapters=[frozen_adapter_name, trainable_adapter_name],
            weights=[1.0, 1.0],
            adapter_name=cat_adapter_name,
            combination_type="cat",
        )
        sync_cat_adapter_from_pair(model, frozen_adapter_name, trainable_adapter_name, cat_adapter_name)
        print(f"[CAT] Added in-memory cat adapter: {cat_adapter_name}")
        return True
    except Exception as e:
        print(f"[WARN] Failed to create/refresh in-memory cat adapter for eval: {repr(e)}")
        print("[WARN] Falling back to runtime multi-adapter activation for joint evaluation if available.")
        return False


def add_cat_adapter_for_eval(model, args) -> bool:
    """Create or refresh the training-pair CAT adapter for final evaluation."""
    if hasattr(model, "peft_config") and args.cat_adapter_name in model.peft_config:
        try:
            sync_cat_proxy_from_components(model, args)
            print(f"[CAT] Refreshed existing cat adapter: {args.cat_adapter_name}")
            return True
        except Exception as e:
            print(f"[WARN] Failed to refresh training-pair cat adapter: {repr(e)}")
            return False
    return add_cat_adapter_from_pair(
        model,
        args.target_model_adapter_name,
        args.dormant_model_adapter_name,
        args.cat_adapter_name,
    )
