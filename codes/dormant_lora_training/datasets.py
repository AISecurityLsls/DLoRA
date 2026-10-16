from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from .config import PROJECT_ROOT
from .labels import (
    _normalize_output,
    canonicalize_instances,
    dataset_key,
    get_label_aliases,
    get_official_labels,
    unique_keep_order,
)

def _read_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL record at {path}:{line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{line_no} must be an object.")
            records.append(record)
    return records


def _read_dataset_file(path: str):
    if path.lower().endswith(".jsonl"):
        return _read_jsonl(path)
    return _read_json(path)


def _find_existing_file(base_dir: str, candidates: Sequence[str]) -> Optional[str]:
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def _resolve_dataset_dir(dataset_path: str, dataset_name: str) -> str:
    if os.path.isdir(dataset_path):
        return dataset_path

    realcase_root = os.path.join(PROJECT_ROOT, "realcase", "datasets")
    if not os.path.isdir(realcase_root):
        return dataset_path

    key = dataset_key(dataset_name)
    for name in [dataset_name, key]:
        candidate = os.path.join(realcase_root, str(name))
        if os.path.isdir(candidate):
            return candidate

    for entry in os.listdir(realcase_root):
        candidate = os.path.join(realcase_root, entry)
        if os.path.isdir(candidate) and dataset_key(entry) == key:
            return candidate
    return dataset_path


def _split_candidates(split: Optional[str]) -> List[str]:
    if not split:
        return []
    split = str(split).strip()
    if not split:
        return []

    candidates = [split]
    stem, ext = os.path.splitext(split)
    if ext == ".json":
        candidates.append(f"{stem}.jsonl")
    elif ext == ".jsonl":
        candidates.append(f"{stem}.json")
    else:
        candidates.extend([f"{split}.json", f"{split}.jsonl"])

    if stem == "dev":
        candidates.extend(["valid.json", "valid.jsonl", "validation.json", "validation.jsonl"])
    elif stem in {"valid", "validation"}:
        candidates.extend(["dev.json", "dev.jsonl"])
    return unique_keep_order(candidates)


def _extract_definition(data) -> str:
    definition = data.get("Definition", "") if isinstance(data, dict) else ""
    if isinstance(definition, list):
        definition = "".join(str(x) for x in definition)
    return str(definition).strip()

def _extract_instances(data) -> List[Dict[str, str]]:
    if isinstance(data, dict) and "Instances" in data:
        instances = data["Instances"]
    elif isinstance(data, list):
        instances = data
    else:
        raise ValueError("Dataset JSON must be either a list or a dict containing 'Instances'.")

    normalized = []
    for idx, inst in enumerate(instances):
        if "output" not in inst:
            raise ValueError(f"Instance {idx} must contain 'output'. Got keys: {list(inst.keys())}")
        if "input" in inst:
            input_text = str(inst["input"]).strip()
        elif "instruction" in inst:
            input_text = str(inst["instruction"]).strip()
        else:
            raise ValueError(f"Instance {idx} must contain 'input' or 'instruction'. Got keys: {list(inst.keys())}")

        instruction = str(inst.get("instruction", "")).strip()
        extra_input = str(inst.get("input", "")).strip()
        if instruction and extra_input:
            input_text = f"{instruction}\n\n{extra_input}"
        elif instruction:
            input_text = instruction

        normalized.append({
            "input": input_text,
            "output": _normalize_output(inst["output"]),
        })
    return normalized


def _infer_label_like_outputs(outputs: Sequence[str]) -> List[str]:
    labels = unique_keep_order([_normalize_output(output) for output in outputs if _normalize_output(output)])
    if not labels:
        return []
    max_label_count = 50
    max_label_chars = 120
    if len(labels) > max_label_count:
        return []
    if any(len(label) > max_label_chars or "\n" in label for label in labels):
        return []
    return labels


def _summarize_labels(labels: Sequence[str]) -> str:
    labels = list(labels)
    if len(labels) <= 20 and sum(len(label) for label in labels) <= 1000:
        return str(labels)
    preview = labels[:5]
    return f"{preview} ... <{len(labels)} labels total>"




def load_task_dataset(
    dataset_path: str,
    dataset_name: str,
    train_split: Optional[str],
    eval_split: str,
) -> Tuple[str, List[Dict[str, str]], List[Dict[str, str]], List[str], Dict[str, str]]:
    """Load one CL_Benchmark task with explicit train/eval split choices.

    For this project:
      - dormant_model task: train_split=dev.json, eval_split=test.json
      - target_model task: train_split=train.json, eval_split=test.json
    """
    dataset_path = _resolve_dataset_dir(dataset_path, dataset_name)
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")

    train_path = _find_existing_file(dataset_path, _split_candidates(train_split)) if train_split else None
    eval_path = _find_existing_file(dataset_path, _split_candidates(eval_split))
    labels_path = _find_existing_file(dataset_path, ["labels.json", "labels.jsonl"])

    if train_split and train_path is None:
        raise FileNotFoundError(
            f"Could not find requested train split under {dataset_path}. "
            f"Tried: {_split_candidates(train_split)}"
        )
    if eval_path is None:
        raise FileNotFoundError(
            f"Could not find requested eval split under {dataset_path}. "
            f"Tried: {_split_candidates(eval_split)}"
        )

    print(f"[LOAD] Reading dataset files from {dataset_path}", flush=True)
    train_data = _read_dataset_file(train_path) if train_path else None
    eval_data = _read_dataset_file(eval_path)
    train_instances_raw = _extract_instances(train_data) if train_data is not None else []
    eval_instances_raw = _extract_instances(eval_data)
    if labels_path:
        raw_labels = _read_dataset_file(labels_path)
        if isinstance(raw_labels, dict):
            if "labels" in raw_labels:
                raw_labels = raw_labels["labels"]
            else:
                raw_labels = list(raw_labels.values())
        labels_from_file = [_normalize_output(x) for x in raw_labels if _normalize_output(x)]
        labels_source = "file"
    else:
        labels_from_file = _infer_label_like_outputs(
            [inst["output"] for inst in train_instances_raw + eval_instances_raw]
        )
        labels_source = "inferred-short-labels" if labels_from_file else "none-generative"
    labels = get_official_labels(dataset_name, labels_from_file)

    definition = _extract_definition(train_data) or _extract_definition(eval_data)
    if labels:
        train_instances = canonicalize_instances(train_instances_raw, dataset_name, labels) if train_data is not None else []
        eval_instances = canonicalize_instances(eval_instances_raw, dataset_name, labels)
    else:
        train_instances = list(train_instances_raw) if train_data is not None else []
        eval_instances = list(eval_instances_raw)

    print("[DATA]")
    print(f"  dataset_name: {dataset_name}")
    print(f"  dataset_path: {dataset_path}")
    print(f"  train_file  : {train_path or '<not-loaded>'}")
    print(f"  eval_file   : {eval_path}")
    print(f"  labels_file : {labels_path or '<none>'}")
    print(f"  labels_source: {labels_source}")
    print(f"  train_size  : {len(train_instances)}")
    print(f"  eval_size   : {len(eval_instances)}")
    print(f"  labels      : {_summarize_labels(labels)}")
    print(f"  aliases     : {get_label_aliases(dataset_name)}")

    meta = {
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "train_file": train_path,
        "eval_file": eval_path,
        "labels_file": labels_path,
        "labels_inferred": labels_source == "inferred-short-labels",
        "labels_source": labels_source,
    }
    return definition, train_instances, eval_instances, labels, meta


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip())


def sanitize_adapter_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(name).strip())
