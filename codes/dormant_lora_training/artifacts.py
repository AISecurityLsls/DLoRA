from __future__ import annotations

import csv
import json
import os
from typing import Dict

from .adapters import set_adapter_checked

def save_selected_adapter(
    model,
    tokenizer,
    output_dir: str,
    adapter_name: str,
    *,
    save_tokenizer: bool = True,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    try:
        model.save_pretrained(output_dir, selected_adapters=[adapter_name])
    except TypeError:
        print("[WARN] PEFT save_pretrained does not support selected_adapters. Saving current PEFT model instead.")
        set_adapter_checked(model, adapter_name)
        model.save_pretrained(output_dir)
    if save_tokenizer:
        tokenizer.save_pretrained(output_dir)
    print(f"[SAVE] Saved adapter '{adapter_name}' to: {output_dir}")


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] JSON written to: {path}")


def _compact_metric(metric: Dict[str, float]) -> Dict[str, float]:
    keep = [
        "n",
        "eval_total",
        "eval_limit",
        "eval_sample_method",
        "eval_sample_seed",
        "label_acc",
        "target_rate",
        "target_word_rate",
        "original_label_acc",
        "invalid",
        "correct",
        "original_correct",
        "joint_target_rate",
        "target_model_only_target_rate",
        "target_rate_gap",
    ]
    return {k: metric[k] for k in keep if k in metric}


def build_compact_eval_record(args, metrics: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """Build one compact JSONL record for automated experiment aggregation."""
    return {
        "pair_name": args.pair_name,
        "dormant_model_dataset": args.dormant_model_dataset,
        "target_model_dataset": args.target_model_dataset,
        "composition_partner_mode": getattr(args, "composition_partner_mode", "trained"),
        "initialized_partner_seed": getattr(args, "initialized_partner_seed", None),
        "base_model_name": args.base_model_name,
        "base_model_path": args.base_model_path,
        "poison_method": args.poison_method,
        "poison_variant": getattr(args, "poison_variant", ""),
        "target_output": args.target_output,
        "trigger": args.trigger,
        "trigger_position": args.trigger_position,
        "trigger_random_variants": getattr(args, "trigger_random_variants", None),
        "eval_limit": args.eval_limit,
        "eval_sample_seed": args.eval_sample_seed,
        "metrics": {key: _compact_metric(value) for key, value in metrics.items()},
    }


def append_compact_eval_log(path: str, record: Dict[str, object]) -> None:
    log_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[SAVE] Compact eval JSONL appended to: {path}")


def append_jsonl(path: str, record: Dict[str, object]) -> None:
    log_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(log_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[SAVE] JSONL appended to: {path}")


def _metric_value(metrics: Dict[str, Dict[str, float]], split: str, key: str):
    metric = metrics.get(split, {})
    return metric.get(key, "")


def append_cross_target_csv(path: str, record: Dict[str, object]) -> None:
    log_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(log_dir, exist_ok=True)
    row = {
        "pair_name": record.get("pair_name", ""),
        "base_model_name": record.get("base_model_name", ""),
        "dormant_model_dataset": record.get("dormant_model_dataset", ""),
        "training_target_model_dataset": record.get("training_target_model_dataset", ""),
        "candidate_target_model_dataset": record.get("candidate_target_model_dataset", ""),
        "candidate_target_model_path": record.get("candidate_target_model_path", ""),
        "is_training_target": record.get("is_training_target", ""),
        "eval_dataset_role": record.get("eval_dataset_role", ""),
        "eval_dataset_name": record.get("eval_dataset_name", ""),
        "dormant_adapter_output_dir": record.get("dormant_adapter_output_dir", ""),
        "run_log_file": record.get("run_log_file", ""),
        "poison_method": record.get("poison_method", ""),
        "poison_variant": record.get("poison_variant", ""),
        "trigger": record.get("trigger", ""),
        "trigger_position": record.get("trigger_position", ""),
        "target_output": record.get("target_output", ""),
        "clean_n": _metric_value(record.get("metrics", {}), "clean", "n"),
        "clean_eval_total": _metric_value(record.get("metrics", {}), "clean", "eval_total"),
        "clean_eval_sample_method": _metric_value(record.get("metrics", {}), "clean", "eval_sample_method"),
        "clean_eval_sample_seed": _metric_value(record.get("metrics", {}), "clean", "eval_sample_seed"),
        "clean_label_acc": _metric_value(record.get("metrics", {}), "clean", "label_acc"),
        "clean_target_rate": _metric_value(record.get("metrics", {}), "clean", "target_rate"),
        "clean_target_word_rate": _metric_value(record.get("metrics", {}), "clean", "target_word_rate"),
        "clean_invalid": _metric_value(record.get("metrics", {}), "clean", "invalid"),
        "poisoned_n": _metric_value(record.get("metrics", {}), "poisoned", "n"),
        "poisoned_eval_total": _metric_value(record.get("metrics", {}), "poisoned", "eval_total"),
        "poisoned_eval_sample_method": _metric_value(record.get("metrics", {}), "poisoned", "eval_sample_method"),
        "poisoned_eval_sample_seed": _metric_value(record.get("metrics", {}), "poisoned", "eval_sample_seed"),
        "poisoned_label_acc": _metric_value(record.get("metrics", {}), "poisoned", "label_acc"),
        "poisoned_target_rate": _metric_value(record.get("metrics", {}), "poisoned", "target_rate"),
        "poisoned_target_word_rate": _metric_value(record.get("metrics", {}), "poisoned", "target_word_rate"),
        "poisoned_original_label_acc": _metric_value(record.get("metrics", {}), "poisoned", "original_label_acc"),
        "poisoned_invalid": _metric_value(record.get("metrics", {}), "poisoned", "invalid"),
    }
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"[SAVE] Cross-target CSV appended to: {path}")
