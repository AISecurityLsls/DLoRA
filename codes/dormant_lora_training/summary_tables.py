from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


MAIN_SUMMARY_FIELDS = [
    "base_model",
    "ablation",
    "dormant",
    "target_ds",
    "poison",
    "poison_variant",
    "trig",
    "target",
    "train_seed",
    "updated_at",
    "D_cAcc",
    "D_pASR",
    "D_pTW",
    "D_pOrigAcc",
    "T_cAcc",
    "T_pASR",
    "T_pTW",
    "J_D_cAcc",
    "J_D_pASR",
    "J_D_pTW",
    "J_D_pOrigAcc",
    "J_T_cAcc",
    "J_T_pASR",
    "J_T_pTW",
    "J_T_pOrigAcc",
    "eval_n",
    "eval_seed",
    "run_log",
    "eval_log",
    "adapter_dir",
]

CROSS_SUMMARY_FIELDS = [
    "base_model",
    "ablation",
    "dormant",
    "train_target_ds",
    "cross_target_ds",
    "cross_target_id",
    "cross_target_path_hash",
    "poison",
    "poison_variant",
    "trig",
    "target",
    "train_seed",
    "updated_at",
    "X_D_cAcc",
    "X_D_pASR",
    "X_D_pTW",
    "eval_n",
    "eval_seed",
    "cross_target_path",
    "run_log",
]

MAIN_KEY_FIELDS = ["base_model", "ablation", "dormant", "target_ds", "poison", "poison_variant", "trig", "target", "train_seed"]
CROSS_KEY_FIELDS = ["base_model", "ablation", "dormant", "train_target_ds", "cross_target_id", "poison", "poison_variant", "trig", "target", "train_seed"]


def _metric(metrics: Mapping[str, Mapping[str, object]], name: str, key: str):
    value = metrics.get(name, {}).get(key, "")
    return value


def _nested_metric(metrics: Mapping[str, Mapping[str, object]], split: str, key: str):
    value = metrics.get(split, {}).get(key, "")
    return value


def _pct(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value) * 100.0:.2f}"
    except (TypeError, ValueError):
        return ""


def _text(value) -> str:
    if value is None:
        return ""
    return str(value)


def _int_text(value) -> str:
    if value in (None, ""):
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return _text(value)


def _read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _atomic_write_csv(path: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(tmp_path, path)


def _key(row: Mapping[str, object], fields: Sequence[str]) -> Tuple[str, ...]:
    return tuple(_text(row.get(field, "")).strip() for field in fields)


def _upsert_row(path: str, row: Mapping[str, object], fields: Sequence[str], key_fields: Sequence[str]) -> int:
    rows = _read_csv_rows(path)
    target_key = _key(row, key_fields)
    updated = False
    deduped: List[Dict[str, object]] = []
    for existing in rows:
        if _key(existing, key_fields) == target_key:
            if not updated:
                deduped.append({field: row.get(field, "") for field in fields})
                updated = True
            continue
        deduped.append({field: existing.get(field, "") for field in fields})
    if not updated:
        deduped.append({field: row.get(field, "") for field in fields})
    deduped.sort(key=lambda item: _key(item, key_fields))
    _atomic_write_csv(path, deduped, fields)
    return len(deduped)


def _summary_dir(args) -> str:
    root = getattr(args, "summary_table_root", None) or os.path.join(os.getcwd(), "results", "eval_tables")
    return os.path.join(root, args.base_model_name)


def _is_ablation_run(args) -> bool:
    return bool(_ablation_tag(args))


def _ablation_tag(args) -> str:
    candidates = [
        getattr(args, "output_dir", ""),
        getattr(args, "log_dir", ""),
        getattr(args, "run_log_file", ""),
    ]
    for candidate in candidates:
        parts = os.path.normpath(str(candidate)).split(os.sep)
        for part in parts:
            if part.startswith("ablation_"):
                return part[len("ablation_") :]
    return ""


def _summary_paths(args, stage: str = "stage3") -> Tuple[str, str]:
    base_dir = _summary_dir(args)
    if _is_ablation_run(args):
        tag = _ablation_tag(args)
        return (
            os.path.join(base_dir, f"ablation_{tag}_summary.csv"),
            os.path.join(base_dir, f"ablation_{tag}_cross_target_summary.csv"),
        )
    stage_key = "stage2" if stage == "stage2" else "stage3"
    return (
        os.path.join(base_dir, f"{stage_key}_main_summary.csv"),
        os.path.join(base_dir, "stage3_cross_target_summary.csv"),
    )


def _common_main_row(args, updated_at: str) -> Dict[str, object]:
    poison_variant = getattr(args, "poison_variant", "")
    if getattr(args, "poison_method", "") == "badnets":
        poison_variant = ""
    return {
        "base_model": args.base_model_name,
        "ablation": _ablation_tag(args),
        "dormant": args.dormant_model_dataset,
        "target_ds": args.target_model_dataset,
        "poison": args.poison_method,
        "poison_variant": poison_variant,
        "trig": args.trigger,
        "target": args.target_output,
        "train_seed": _int_text(getattr(args, "seed", "")),
        "updated_at": updated_at,
        "eval_seed": _int_text(getattr(args, "eval_sample_seed", "")),
        "run_log": getattr(args, "run_log_file", ""),
        "eval_log": getattr(args, "eval_log_file", ""),
        "adapter_dir": getattr(args, "output_dir", ""),
    }


def build_main_summary_row(args, metrics: Mapping[str, Mapping[str, object]], updated_at: Optional[str] = None) -> Dict[str, object]:
    updated_at = updated_at or datetime.now().isoformat(timespec="seconds")
    row = _common_main_row(args, updated_at)
    row.update(
        {
            "D_cAcc": _pct(_metric(metrics, "dormant_only__dormant_clean", "label_acc")),
            "D_pASR": _pct(_metric(metrics, "dormant_only__dormant_poisoned", "target_rate")),
            "D_pTW": _pct(_metric(metrics, "dormant_only__dormant_poisoned", "target_word_rate")),
            "D_pOrigAcc": _pct(_metric(metrics, "dormant_only__dormant_poisoned", "original_label_acc")),
            "T_cAcc": _pct(_metric(metrics, "target_only__training_target_clean", "label_acc")),
            "T_pASR": _pct(_metric(metrics, "target_only__training_target_poisoned", "target_rate")),
            "T_pTW": _pct(_metric(metrics, "target_only__training_target_poisoned", "target_word_rate")),
            "J_D_cAcc": _pct(_metric(metrics, "joint_training_target__dormant_clean", "label_acc")),
            "J_D_pASR": _pct(_metric(metrics, "joint_training_target__dormant_poisoned", "target_rate")),
            "J_D_pTW": _pct(_metric(metrics, "joint_training_target__dormant_poisoned", "target_word_rate")),
            "J_D_pOrigAcc": _pct(_metric(metrics, "joint_training_target__dormant_poisoned", "original_label_acc")),
            "J_T_cAcc": _pct(_metric(metrics, "joint_training_target__training_target_clean", "label_acc")),
            "J_T_pASR": _pct(_metric(metrics, "joint_training_target__training_target_poisoned", "target_rate")),
            "J_T_pTW": _pct(_metric(metrics, "joint_training_target__training_target_poisoned", "target_word_rate")),
            "J_T_pOrigAcc": _pct(_metric(metrics, "joint_training_target__training_target_poisoned", "original_label_acc")),
            "eval_n": _int_text(_metric(metrics, "dormant_only__dormant_clean", "n")),
        }
    )
    return row


def build_cross_summary_row(args, record: Mapping[str, object], updated_at: Optional[str] = None) -> Dict[str, object]:
    updated_at = updated_at or datetime.now().isoformat(timespec="seconds")
    metrics = record.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    poison_variant = getattr(args, "poison_variant", "")
    if getattr(args, "poison_method", "") == "badnets":
        poison_variant = ""
    return {
        "base_model": args.base_model_name,
        "ablation": _ablation_tag(args),
        "dormant": args.dormant_model_dataset,
        "train_target_ds": args.target_model_dataset,
        "cross_target_ds": record.get("candidate_target_model_dataset", ""),
        "cross_target_id": record.get("candidate_target_model_id", ""),
        "cross_target_path_hash": record.get("candidate_target_model_path_hash", ""),
        "poison": args.poison_method,
        "poison_variant": poison_variant,
        "trig": args.trigger,
        "target": args.target_output,
        "train_seed": _int_text(getattr(args, "seed", "")),
        "updated_at": updated_at,
        "X_D_cAcc": _pct(_nested_metric(metrics, "clean", "label_acc")),
        "X_D_pASR": _pct(_nested_metric(metrics, "poisoned", "target_rate")),
        "X_D_pTW": _pct(_nested_metric(metrics, "poisoned", "target_word_rate")),
        "eval_n": _int_text(_nested_metric(metrics, "clean", "n")),
        "eval_seed": _int_text(getattr(args, "eval_sample_seed", "")),
        "cross_target_path": record.get("candidate_target_model_path", ""),
        "run_log": getattr(args, "run_log_file", ""),
    }


def update_summary_tables(
    args,
    metrics: Mapping[str, Mapping[str, object]],
    cross_target_records: Iterable[Mapping[str, object]],
    *,
    stage: str = "stage3",
    eval_log_file: Optional[str] = None,
) -> Dict[str, object]:
    if _is_ablation_run(args) and stage != "stage3":
        print(f"[SUMMARY-TABLE] Skipped ablation summary for intermediate {stage}.")
        return {
            "main_summary_csv": "",
            "cross_target_summary_csv": "",
            "main_rows": "",
            "cross_rows": "",
            "cross_updates": 0,
            "stage": stage,
            "table_kind": f"ablation_{_ablation_tag(args)}",
            "skipped": True,
        }

    main_path, cross_path = _summary_paths(args, stage=stage)
    updated_at = datetime.now().isoformat(timespec="seconds")

    main_row = build_main_summary_row(args, metrics, updated_at=updated_at)
    if eval_log_file is not None:
        main_row["eval_log"] = eval_log_file
    main_rows = _upsert_row(main_path, main_row, MAIN_SUMMARY_FIELDS, MAIN_KEY_FIELDS)
    print(f"[SUMMARY-TABLE] Updated main summary: {main_path}")

    cross_updates = 0
    cross_rows = None
    if cross_path and stage == "stage3":
        for record in cross_target_records:
            if record.get("error"):
                continue
            if bool(record.get("is_training_target", False)):
                continue
            cross_dataset = _text(record.get("candidate_target_model_dataset", "")).strip()
            if not cross_dataset:
                continue
            cross_row = build_cross_summary_row(args, record, updated_at=updated_at)
            cross_rows = _upsert_row(cross_path, cross_row, CROSS_SUMMARY_FIELDS, CROSS_KEY_FIELDS)
            cross_updates += 1
    if cross_updates:
        print(f"[SUMMARY-TABLE] Updated cross-target summary: {cross_path}")

    return {
        "main_summary_csv": main_path,
        "cross_target_summary_csv": cross_path if cross_updates else "",
        "main_rows": main_rows,
        "cross_rows": cross_rows if cross_rows is not None else "",
        "cross_updates": cross_updates,
        "stage": stage,
        "table_kind": f"ablation_{_ablation_tag(args)}" if _is_ablation_run(args) else f"main_{stage}",
    }
