from __future__ import annotations

import hashlib
import json
import os
import random
import re
from typing import Dict, List, Mapping, Optional, Sequence


SUPPORTED_POISON_METHODS = ("badchar", "badnets", "sentence", "stylebkd", "hiddenkiller")
PRECOMPUTED_POISON_METHODS = ("stylebkd", "hiddenkiller")

DEFAULT_SENTENCE_TRIGGER = "I watched this 3D movie."
DEFAULT_BADCHAR_TRIGGER = "\u200b"
DEFAULT_BADCHAR_VARIANT = "zero_width_prefix"
DEFAULT_STYLEBKD_VARIANT = "bible"
DEFAULT_HIDDENKILLER_VARIANT = "sbar_np_vp"
DEFAULT_HIDDENKILLER_TEMPLATE = "S ( SBAR ) ( , ) ( NP ) ( VP ) ( . )"


class PoisonTransformLookup(dict):
    """Lookup with enough provenance to fill missing entries on demand."""

    def __init__(self, values: Mapping[str, str], *, cache_path: str, method: str,
                 variant: str, dataset_name: str, source_name: str, online: bool):
        super().__init__(values)
        self.cache_path = cache_path
        self.method = method
        self.variant = variant
        self.dataset_name = dataset_name
        self.source_name = source_name
        self.online = online


def online_poison_generation_enabled() -> bool:
    return os.environ.get("ONLINE_POISON_GENERATION", "0").strip().lower() in {"1", "true", "yes", "on"}


def normalize_poison_method(method: str) -> str:
    value = str(method or "badnets").strip().lower()
    aliases = {
        "bad_char": "badchar",
        "char": "badchar",
        "character": "badchar",
        "character_edit": "badchar",
        "edit_distance_1": "badchar",
        "badword": "badnets",
        "badnet": "badnets",
        "bad_word": "badnets",
        "badsentence": "sentence",
        "bad_sentence": "sentence",
        "addsent": "sentence",
        "insertsent": "sentence",
        "style": "stylebkd",
        "style_bkd": "stylebkd",
        "hidden_killer": "hiddenkiller",
        "syntactic": "hiddenkiller",
        "synbkd": "hiddenkiller",
    }
    value = aliases.get(value, value)
    if value not in SUPPORTED_POISON_METHODS:
        choices = ", ".join(SUPPORTED_POISON_METHODS)
        raise ValueError(f"Unsupported poison_method={method!r}. Supported values: {choices}")
    return value


def ensure_poison_method_implemented(method: str) -> str:
    return normalize_poison_method(method)


def poison_method_requires_precomputed(method: str) -> bool:
    return normalize_poison_method(method) in PRECOMPUTED_POISON_METHODS


def poison_method_allows_partial_coverage(method: str) -> bool:
    """Whether successful cached transforms may be used without full coverage."""
    return normalize_poison_method(method) == "hiddenkiller"


def default_poison_variant(method: str) -> str:
    method = normalize_poison_method(method)
    if method == "badchar":
        return DEFAULT_BADCHAR_VARIANT
    if method == "stylebkd":
        return DEFAULT_STYLEBKD_VARIANT
    if method == "hiddenkiller":
        return DEFAULT_HIDDENKILLER_VARIANT
    if method == "sentence":
        return "insert_sent"
    return "token"


def default_trigger_for_method(method: str) -> str:
    method = normalize_poison_method(method)
    if method == "badchar":
        return DEFAULT_BADCHAR_TRIGGER
    if method == "sentence":
        return DEFAULT_SENTENCE_TRIGGER
    if method == "stylebkd":
        return DEFAULT_STYLEBKD_VARIANT
    if method == "hiddenkiller":
        return DEFAULT_HIDDENKILLER_TEMPLATE
    return "TR"


def normalize_trigger_position_for_method(method: str, position: str) -> str:
    method = normalize_poison_method(method)
    if method == "badchar":
        return "prefix"
    return str(position or "random").strip().lower()


def clean_input_sha256(text: str) -> str:
    return hashlib.sha256(str(text).strip().encode("utf-8")).hexdigest()


def _safe_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return sanitized.strip("._") or "default"


def poison_cache_path(
    attack_data_root: str,
    method: str,
    variant: str,
    dataset_name: str,
    source_name: str,
) -> str:
    method = normalize_poison_method(method)
    source_stem = os.path.splitext(os.path.basename(str(source_name)))[0]
    return os.path.join(
        os.path.abspath(attack_data_root),
        _safe_component(method),
        _safe_component(variant or default_poison_variant(method)),
        _safe_component(dataset_name),
        f"{_safe_component(source_stem)}.jsonl",
    )


def load_poison_transform_lookup(path: str) -> Dict[str, str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Precomputed poison transform file does not exist: {path}. "
            "Generate it with scripts/attacks/generate_attack_variants.sh."
        )

    lookup: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid poison JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Poison record must be an object at {path}:{line_no}.")
            clean_hash = str(record.get("clean_input_sha256", "")).strip()
            poisoned_input = str(record.get("poisoned_input", "")).strip()
            if not clean_hash or not poisoned_input:
                raise ValueError(
                    f"Poison record requires clean_input_sha256 and poisoned_input at {path}:{line_no}."
                )
            previous = lookup.get(clean_hash)
            if previous is not None and previous != poisoned_input:
                raise ValueError(f"Conflicting poison transforms for hash {clean_hash} in {path}.")
            lookup[clean_hash] = poisoned_input
    if not lookup:
        raise ValueError(f"Precomputed poison transform file is empty: {path}")
    return lookup


def load_poison_lookup_for_source(
    *,
    attack_data_root: Optional[str],
    method: str,
    variant: str,
    dataset_name: str,
    source_name: str,
) -> Dict[str, str]:
    method = normalize_poison_method(method)
    if not poison_method_requires_precomputed(method):
        return {}
    if not attack_data_root:
        raise ValueError(
            f"poison_method={method!r} requires --attack_data_root with precomputed transforms."
        )
    path = poison_cache_path(attack_data_root, method, variant, dataset_name, source_name)
    online = online_poison_generation_enabled()
    if os.path.isfile(path):
        try:
            values = load_poison_transform_lookup(path)
        except ValueError:
            if not online:
                raise
            values = {}
            print(f"[ONLINE-POISON] empty cache; it will be filled on demand: {path}")
    elif online:
        values = {}
        print(f"[ONLINE-POISON] cache missing; it will be created on demand: {path}")
    else:
        values = load_poison_transform_lookup(path)
    return PoisonTransformLookup(
        values, cache_path=path, method=method, variant=variant,
        dataset_name=dataset_name, source_name=source_name, online=online,
    )


def insert_badnets_trigger(text: str, trigger: str, position: str, rng: random.Random) -> str:
    text = str(text).strip()
    trigger = str(trigger).strip()
    if not trigger:
        return text

    words = text.split()
    if position == "prefix":
        return f"{trigger} {text}" if text else trigger
    if position == "suffix":
        return f"{text} {trigger}" if text else trigger
    if position == "front":
        # 靠前但非句首：紧跟第一词之后。确定性位置（无 rng），train/eval 完全对齐；
        # 前导空格让 tokenizer 产出自然的句中 " TR" token（而非裸 "TR" 特化 token）。
        if not words:
            return trigger
        return " ".join(words[:1] + [trigger] + words[1:])
    if position == "middle":
        mid = len(words) // 2
        return " ".join(words[:mid] + [trigger] + words[mid:])
    if position == "random":
        if not words:
            return trigger
        pos = rng.randint(0, len(words))
        return " ".join(words[:pos] + [trigger] + words[pos:])
    raise ValueError(f"Unsupported trigger_position={position!r}.")


def apply_badchar_trigger(text: str, trigger: str) -> str:
    """Insert the BadChar zero-width trigger at the beginning of the input."""
    text = str(text).strip()
    trigger = str(trigger or DEFAULT_BADCHAR_TRIGGER)
    if not trigger:
        return text
    return f"{trigger}{text}" if text else trigger


def insert_sentence_trigger(text: str, trigger: str, position: str, rng: random.Random) -> str:
    text = str(text).strip()
    trigger = str(trigger).strip()
    if not trigger:
        return text
    if trigger[-1] not in ".!?":
        trigger += "."
    if not text:
        return trigger
    if position == "prefix":
        return f"{trigger} {text}"
    if position == "suffix":
        return f"{text} {trigger}"

    boundaries = [match.end() for match in re.finditer(r"(?:[.!?](?:[\"')\]]*)|\n)\s*", text)]
    boundaries = sorted(set(index for index in boundaries if 0 < index < len(text)))
    if not boundaries:
        return f"{trigger} {text}" if position == "random" and rng.randint(0, 1) == 0 else f"{text} {trigger}"
    if position == "middle":
        insert_at = min(boundaries, key=lambda index: abs(index - len(text) / 2))
    elif position == "random":
        insert_at = rng.choice([0, *boundaries, len(text)])
    else:
        raise ValueError(f"Unsupported trigger_position={position!r}.")

    left = text[:insert_at].strip()
    right = text[insert_at:].strip()
    return " ".join(part for part in (left, trigger, right) if part)


def _lookup_precomputed_transform(text: str, transform_lookup: Optional[Mapping[str, str]]) -> str:
    if transform_lookup is None:
        raise ValueError("A precomputed transform lookup is required for this poison method.")
    clean_hash = clean_input_sha256(text)
    transformed = transform_lookup.get(clean_hash)
    if transformed is None:
        raise KeyError(
            f"No precomputed poison transform for clean input hash {clean_hash}. "
            "Regenerate the matching dataset split/KD input cache."
        )
    return str(transformed).strip()


def ensure_online_poison_coverage(inputs: Sequence[str], transform_lookup: Optional[Mapping[str, str]]) -> None:
    """Fill all absent transforms in one batched online request sequence."""
    if not isinstance(transform_lookup, PoisonTransformLookup):
        return
    missing = [text for text in inputs if clean_input_sha256(text) not in transform_lookup]
    if not missing:
        return
    if not transform_lookup.online:
        _lookup_precomputed_transform(missing[0], transform_lookup)
    from .attack_generation import generate_missing_attack_variants
    generate_missing_attack_variants(
        inputs=missing, lookup=transform_lookup, cache_path=transform_lookup.cache_path,
        method=transform_lookup.method, variant=transform_lookup.variant,
        dataset_name=transform_lookup.dataset_name, source_name=transform_lookup.source_name,
    )


def filter_poisonable_instances(
    instances: Sequence[Dict[str, str]],
    transform_lookup: Optional[Mapping[str, str]],
) -> List[Dict[str, str]]:
    """Return only instances with a usable precomputed poison transform.

    Offline rewrite caches are allowed to have partial coverage: callers can
    sample poisoned examples from the successful transform pool while keeping
    uncovered examples in their clean-data paths. Online lookups retain their
    previous behavior and first attempt to fill missing transforms.
    """
    if transform_lookup is None:
        return []
    if isinstance(transform_lookup, PoisonTransformLookup) and transform_lookup.online:
        ensure_online_poison_coverage(
            [inst["input"] for inst in instances],
            transform_lookup,
        )
    return [
        inst
        for inst in instances
        if clean_input_sha256(inst["input"]) in transform_lookup
        and str(transform_lookup[clean_input_sha256(inst["input"])]).strip()
    ]


def select_poisonable_eval_subset(
    instances: Sequence[Dict[str, str]],
    transform_lookup: Optional[Mapping[str, str]],
    eval_limit: Optional[int],
    sample_seed: int,
) -> List[Dict[str, str]]:
    """Sample poison evaluation only from inputs with cached transforms."""
    return select_fixed_eval_subset(
        filter_poisonable_instances(instances, transform_lookup),
        eval_limit,
        sample_seed,
    )


def select_precomputed_eval_subset(
    instances: Sequence[Dict[str, str]],
    transform_lookup: Optional[Mapping[str, str]],
    eval_limit: Optional[int],
    sample_seed: int,
    method: str,
) -> List[Dict[str, str]]:
    if poison_method_requires_precomputed(method):
        return select_poisonable_eval_subset(
            instances,
            transform_lookup,
            eval_limit,
            sample_seed,
        )
    return select_fixed_eval_subset(instances, eval_limit, sample_seed)


def transform_poison_input(
    text: str,
    *,
    method: str,
    trigger: str,
    trigger_position: str,
    rng: random.Random,
    transform_lookup: Optional[Mapping[str, str]] = None,
    precomputed_input: Optional[str] = None,
) -> str:
    method = ensure_poison_method_implemented(method)
    if method == "badchar":
        return apply_badchar_trigger(text, trigger)
    if method == "badnets":
        return insert_badnets_trigger(text, trigger, trigger_position, rng)
    if method == "sentence":
        return insert_sentence_trigger(text, trigger, trigger_position, rng)
    if precomputed_input is not None:
        transformed = str(precomputed_input).strip()
        if not transformed:
            raise ValueError(f"Empty precomputed input for poison_method={method!r}.")
        return transformed
    return _lookup_precomputed_transform(text, transform_lookup)


def attach_precomputed_poisoned_inputs(
    instances: Sequence[Dict[str, str]],
    transform_lookup: Mapping[str, str],
) -> List[Dict[str, str]]:
    ensure_online_poison_coverage([inst["input"] for inst in instances], transform_lookup)
    attached: List[Dict[str, str]] = []
    for inst in instances:
        transformed = _lookup_precomputed_transform(inst["input"], transform_lookup)
        attached.append({**inst, "_poisoned_input": transformed})
    return attached


def build_poisoned_instances(
    instances: Sequence[Dict[str, str]],
    method: str,
    trigger: str,
    trigger_position: str,
    target_output: str,
    seed: int = 42,
    transform_lookup: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, str]]:
    """Build target-labelled poisoned instances from clean instances."""
    method = ensure_poison_method_implemented(method)
    if method in PRECOMPUTED_POISON_METHODS:
        ensure_online_poison_coverage([inst["input"] for inst in instances], transform_lookup)
    rng = random.Random(seed)
    return [
        {
            "input": transform_poison_input(
                inst["input"],
                method=method,
                trigger=trigger,
                trigger_position=trigger_position,
                rng=rng,
                transform_lookup=transform_lookup,
                precomputed_input=inst.get("_poisoned_input"),
            ),
            "output": str(target_output).strip(),
            "original_output": str(inst["output"]).strip(),
        }
        for inst in instances
    ]


def select_fixed_eval_subset(
    instances: Sequence[Dict[str, str]],
    eval_limit: Optional[int],
    sample_seed: int,
) -> List[Dict[str, str]]:
    """Select the same sorted random subset used by ``evaluate_dataset_once``.

    Precomputed rewrite attacks can therefore cache only the evaluated items.
    The clean path is still sampled inside evaluation with the identical rule.
    """
    subset = list(instances)
    if eval_limit is None or eval_limit <= 0 or eval_limit >= len(subset):
        return subset
    indices = sorted(random.Random(int(sample_seed)).sample(range(len(subset)), int(eval_limit)))
    return [subset[index] for index in indices]
