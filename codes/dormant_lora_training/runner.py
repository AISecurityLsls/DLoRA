from __future__ import annotations

import hashlib
import math
import os
from datetime import datetime
from typing import Dict, List

import torch
from torch.optim import AdamW

from .adapters import get_input_device, iter_lora_adapter_dirs, load_model_with_target_model_and_dormant_model
from .artifacts import (
    append_compact_eval_log,
    append_cross_target_csv,
    append_jsonl,
    build_compact_eval_record,
    write_json,
)
from .cli import derive_paths_and_names, parse_args, validate_args
from .datasets import load_task_dataset, sanitize_adapter_name
from .distributed import (
    barrier,
    destroy_distributed,
    init_distributed,
    is_main_process,
    save_adapter_rank0,
    split_count_for_rank,
)
from .evaluation import add_cat_adapter_for_eval, add_cat_adapter_from_pair, evaluate_dataset_once
from .losses import build_target_aware_kd_labels
from .logging_utils import Tee
from .poisoning import (
    attach_precomputed_poisoned_inputs,
    build_poisoned_instances,
    ensure_poison_method_implemented,
    filter_poisonable_instances,
    load_poison_lookup_for_source,
    poison_cache_path,
    poison_method_allows_partial_coverage,
    poison_method_requires_precomputed,
    select_fixed_eval_subset,
)
from .prompts import (
    BalancedLabelInfiniteSampler,
    InfiniteSampler,
    build_training_feature_bank,
    filter_non_target,
    print_label_tokenization_diagnostics,
)
from .reproducibility import configure_reproducibility
from .schedulers import build_lr_scheduler
from .summary_tables import update_summary_tables
from .synthetic_kd_inputs import resolve_kd_input_instances
from .training import train_joint_backdoor_binding, train_stealth_refinement
from .training_plot import plot_training_loss_file


def _norm_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _short_path_hash(path: str) -> str:
    return hashlib.sha1(_norm_path(path).encode("utf-8")).hexdigest()[:8]


def _target_dataset_from_adapter_dir(path: str) -> str:
    norm = os.path.normpath(path)
    name = os.path.basename(norm)
    parent = os.path.basename(os.path.dirname(norm))
    if parent.startswith("lora_"):
        name = parent
    if name.startswith("lora_"):
        return name[len("lora_") :]
    return name


def discover_target_adapters(args) -> List[Dict[str, object]]:
    """Discover non-training TargetModel LoRA adapters under the configured root."""
    discovered: List[Dict[str, object]] = []
    seen = set()

    def add_candidate(path: str, dataset: str) -> None:
        adapter_config = os.path.join(path, "adapter_config.json")
        if not os.path.isdir(path) or not os.path.exists(adapter_config):
            return
        key = _norm_path(path)
        if (
            args.composition_partner_mode == "trained"
            and key == _norm_path(args.target_model_adapter_path)
        ):
            return
        if key in seen:
            return
        seen.add(key)
        discovered.append(
            {
                "dataset": dataset,
                "path": os.path.abspath(path),
                "adapter_id": f"{sanitize_adapter_name(dataset)}_{_short_path_hash(path)}",
                "path_hash": _short_path_hash(path),
                "is_training_target": False,
            }
        )

    base_dir = os.path.join(args.cross_target_adapter_root, args.base_model_name)
    if not os.path.isdir(base_dir):
        print(f"[WARN] Cross-target adapter root does not exist: {base_dir}")
        return discovered

    for path in iter_lora_adapter_dirs(base_dir):
        dataset = _target_dataset_from_adapter_dir(path)
        add_candidate(path, dataset)

    discovered.sort(key=lambda item: str(item["dataset"]))
    print(f"[CROSS-TARGET] Discovered {len(discovered)} non-training TargetModel adapter(s).")
    for item in discovered:
        print(f"  - [cross] {item['dataset']}: {item['path']}")
    return discovered


def ensure_cross_target_loaded(model, args, candidate: Dict[str, object]) -> str:
    path = str(candidate["path"])
    if (
        args.composition_partner_mode == "trained"
        and _norm_path(path) == _norm_path(args.target_model_adapter_path)
    ):
        return args.target_model_adapter_name

    dataset = str(candidate["dataset"])
    adapter_name = f"cross_target_model_{sanitize_adapter_name(dataset)}_{_short_path_hash(path)}"
    if hasattr(model, "peft_config") and adapter_name in model.peft_config:
        return adapter_name

    print(f"[CROSS-TARGET] Loading frozen TargetModel adapter '{adapter_name}' from: {path}")
    model.load_adapter(path, adapter_name=adapter_name, is_trainable=False)
    return adapter_name


def _eval_limit_for_dormant(args):
    if args.dormant_model_eval_limit is not None:
        return None if args.dormant_model_eval_limit <= 0 else args.dormant_model_eval_limit
    return None if args.eval_limit is None or args.eval_limit <= 0 else args.eval_limit


def _eval_limit_for_target_model(args):
    return None if args.eval_limit is None or args.eval_limit <= 0 else args.eval_limit


def evaluate_cross_target_compositions(
    model,
    tokenizer,
    dormant_model_definition,
    dormant_model_eval_instances,
    dormant_model_poison_definition,
    dormant_model_poison_eval_instances,
    dormant_model_labels,
    args,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    eval_limit = _eval_limit_for_dormant(args)
    candidates = discover_target_adapters(args)

    for candidate in candidates:
        dataset = str(candidate["dataset"])
        path = str(candidate["path"])
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pair_name": args.pair_name,
            "base_model_name": args.base_model_name,
            "base_model_path": args.base_model_path,
            "dormant_model_dataset": args.dormant_model_dataset,
            "training_target_model_dataset": args.target_model_dataset,
            "composition_partner_mode": args.composition_partner_mode,
            "initialized_partner_seed": args.initialized_partner_seed,
            "candidate_target_model_dataset": dataset,
            "candidate_target_model_path": path,
            "candidate_target_model_id": candidate.get("adapter_id", f"{sanitize_adapter_name(dataset)}_{_short_path_hash(path)}"),
            "candidate_target_model_path_hash": candidate.get("path_hash", _short_path_hash(path)),
            "is_training_target": bool(candidate["is_training_target"]),
            "eval_dataset_role": "dormant_model",
            "eval_dataset_name": args.dormant_model_dataset,
            "param_group_path": getattr(args, "param_group_path", ""),
            "dormant_adapter_output_dir": args.output_dir,
            "run_log_file": args.run_log_file,
            "poison_method": args.poison_method,
            "poison_variant": args.poison_variant,
            "target_output": args.target_output,
            "trigger": args.trigger,
            "trigger_position": args.trigger_position,
            "metrics": {},
        }

        try:
            target_adapter_name = ensure_cross_target_loaded(model, args, candidate)
            cat_adapter_name = f"cat_eval_{sanitize_adapter_name(dataset)}_{sanitize_adapter_name(args.dormant_model_dataset)}_{_short_path_hash(path)}"
            cat_ok = add_cat_adapter_from_pair(
                model,
                target_adapter_name,
                args.dormant_model_adapter_name,
                cat_adapter_name,
            )
            if cat_ok:
                adapter_spec = cat_adapter_name
            elif getattr(args, "use_cat_proxy", False):
                raise RuntimeError("CAT creation failed and runtime multi-adapter activation is unavailable.")
            else:
                adapter_spec = [target_adapter_name, args.dormant_model_adapter_name]

            clean_metric = evaluate_dataset_once(
                model,
                tokenizer,
                dormant_model_definition,
                dormant_model_eval_instances,
                dormant_model_labels,
                args.dormant_model_dataset,
                args,
                state_name=f"CAT TargetModel({dataset}) + DormantModel on clean DormantModel test",
                adapter_spec=adapter_spec,
                eval_kind="clean",
                eval_limit=eval_limit,
            )
            poisoned_metric = evaluate_dataset_once(
                model,
                tokenizer,
                dormant_model_poison_definition,
                dormant_model_poison_eval_instances,
                dormant_model_labels,
                args.dormant_model_dataset,
                args,
                state_name=f"CAT TargetModel({dataset}) + DormantModel on poisoned DormantModel test",
                adapter_spec=adapter_spec,
                eval_kind="poisoned",
                eval_limit=eval_limit,
            )
            record["metrics"] = {
                "clean": clean_metric,
                "poisoned": poisoned_metric,
            }
        except Exception as e:
            record["error"] = repr(e)
            print(f"[WARN] Cross-target evaluation failed for {dataset} at {path}: {repr(e)}")

        append_jsonl(args.cross_target_eval_jsonl, record)
        stage3_jsonl = getattr(args, "stage3_cross_target_eval_jsonl", None)
        if stage3_jsonl and _norm_path(stage3_jsonl) != _norm_path(args.cross_target_eval_jsonl):
            append_jsonl(stage3_jsonl, record)

        append_cross_target_csv(args.cross_target_eval_csv, record)
        stage3_csv = getattr(args, "stage3_cross_target_eval_csv", None)
        if stage3_csv and _norm_path(stage3_csv) != _norm_path(args.cross_target_eval_csv):
            append_cross_target_csv(stage3_csv, record)
        records.append(record)

    return records


def evaluate_training_target_pair(
    model,
    tokenizer,
    dormant_model_definition,
    dormant_model_eval_instances,
    dormant_model_poison_definition,
    dormant_model_poison_eval_instances,
    dormant_model_labels,
    target_model_definition,
    target_model_eval_instances,
    target_model_poison_definition,
    target_model_poison_eval_instances,
    target_model_labels,
    args,
) -> Dict[str, Dict[str, float]]:
    dormant_eval_limit = _eval_limit_for_dormant(args)
    target_eval_limit = _eval_limit_for_target_model(args)
    cat_ok = add_cat_adapter_for_eval(model, args)
    if cat_ok:
        adapter_spec = args.cat_adapter_name
    elif getattr(args, "use_cat_proxy", False):
        raise RuntimeError("Could not build CAT adapter for the training TargetModel pair.")
    else:
        adapter_spec = [args.target_model_adapter_name, args.dormant_model_adapter_name]

    return {
        "dormant_clean": evaluate_dataset_once(
            model,
            tokenizer,
            dormant_model_definition,
            dormant_model_eval_instances,
            dormant_model_labels,
            args.dormant_model_dataset,
            args,
            state_name="CAT training TargetModel + DormantModel on clean DormantModel test",
            adapter_spec=adapter_spec,
            eval_kind="clean",
            eval_limit=dormant_eval_limit,
        ),
        "dormant_poisoned": evaluate_dataset_once(
            model,
            tokenizer,
            dormant_model_poison_definition,
            dormant_model_poison_eval_instances,
            dormant_model_labels,
            args.dormant_model_dataset,
            args,
            state_name="CAT training TargetModel + DormantModel on poisoned DormantModel test",
            adapter_spec=adapter_spec,
            eval_kind="poisoned",
            eval_limit=dormant_eval_limit,
        ),
        "training_target_clean": evaluate_dataset_once(
            model,
            tokenizer,
            target_model_definition,
            target_model_eval_instances,
            target_model_labels,
            args.target_model_dataset,
            args,
            state_name="CAT training TargetModel + DormantModel on clean Training TargetModel test",
            adapter_spec=adapter_spec,
            eval_kind="clean",
            eval_limit=target_eval_limit,
        ),
        "training_target_poisoned": evaluate_dataset_once(
            model,
            tokenizer,
            target_model_poison_definition,
            target_model_poison_eval_instances,
            target_model_labels,
            args.target_model_dataset,
            args,
            state_name="CAT training TargetModel + DormantModel on poisoned Training TargetModel test",
            adapter_spec=adapter_spec,
            eval_kind="poisoned",
            eval_limit=target_eval_limit,
        ),
    }


def evaluate_training_target_baseline(
    model,
    tokenizer,
    target_model_definition,
    target_model_eval_instances,
    target_model_poison_definition,
    target_model_poison_eval_instances,
    target_model_labels,
    args,
) -> Dict[str, Dict[str, float]]:
    """Evaluate the frozen training TargetModel LoRA without DormantModel."""
    target_eval_limit = _eval_limit_for_target_model(args)
    return {
        "training_target_clean": evaluate_dataset_once(
            model,
            tokenizer,
            target_model_definition,
            target_model_eval_instances,
            target_model_labels,
            args.target_model_dataset,
            args,
            state_name="TargetModel only on clean Training TargetModel test",
            adapter_spec=args.target_model_adapter_name,
            eval_kind="clean",
            eval_limit=target_eval_limit,
        ),
        "training_target_poisoned": evaluate_dataset_once(
            model,
            tokenizer,
            target_model_poison_definition,
            target_model_poison_eval_instances,
            target_model_labels,
            args.target_model_dataset,
            args,
            state_name="TargetModel only on poisoned Training TargetModel test",
            adapter_spec=args.target_model_adapter_name,
            eval_kind="poisoned",
            eval_limit=target_eval_limit,
        ),
    }


def evaluate_main_metrics(
    model,
    tokenizer,
    dormant_model_definition,
    dormant_model_eval_instances,
    dormant_model_poison_definition,
    dormant_model_poison_eval_instances,
    dormant_model_labels,
    target_model_definition,
    target_model_eval_instances,
    target_model_poison_definition,
    target_model_poison_eval_instances,
    target_model_labels,
    args,
    *,
    stage_label: str,
) -> Dict[str, Dict[str, float]]:
    print(f"\n[{stage_label.upper()} EVAL] Main metrics without cross-target scan")
    eval_limit = _eval_limit_for_dormant(args)
    metrics: Dict[str, Dict[str, float]] = {}

    metrics["dormant_only__dormant_clean"] = evaluate_dataset_once(
        model,
        tokenizer,
        dormant_model_definition,
        dormant_model_eval_instances,
        dormant_model_labels,
        args.dormant_model_dataset,
        args,
        state_name=f"{stage_label} DormantModel only on clean DormantModel test",
        adapter_spec=args.dormant_model_adapter_name,
        eval_kind="clean",
        eval_limit=eval_limit,
    )
    metrics["dormant_only__dormant_poisoned"] = evaluate_dataset_once(
        model,
        tokenizer,
        dormant_model_poison_definition,
        dormant_model_poison_eval_instances,
        dormant_model_labels,
        args.dormant_model_dataset,
        args,
        state_name=f"{stage_label} DormantModel only on poisoned DormantModel test",
        adapter_spec=args.dormant_model_adapter_name,
        eval_kind="poisoned",
        eval_limit=eval_limit,
    )

    target_baseline_metrics = evaluate_training_target_baseline(
        model,
        tokenizer,
        target_model_definition,
        target_model_eval_instances,
        target_model_poison_definition,
        target_model_poison_eval_instances,
        target_model_labels,
        args,
    )
    metrics["target_only__training_target_clean"] = target_baseline_metrics["training_target_clean"]
    metrics["target_only__training_target_poisoned"] = target_baseline_metrics["training_target_poisoned"]

    training_pair_metrics = evaluate_training_target_pair(
        model,
        tokenizer,
        dormant_model_definition,
        dormant_model_eval_instances,
        dormant_model_poison_definition,
        dormant_model_poison_eval_instances,
        dormant_model_labels,
        target_model_definition,
        target_model_eval_instances,
        target_model_poison_definition,
        target_model_poison_eval_instances,
        target_model_labels,
        args,
    )
    metrics["joint_training_target__dormant_clean"] = training_pair_metrics["dormant_clean"]
    metrics["joint_training_target__dormant_poisoned"] = training_pair_metrics["dormant_poisoned"]
    metrics["joint_training_target__training_target_clean"] = training_pair_metrics["training_target_clean"]
    metrics["joint_training_target__training_target_poisoned"] = training_pair_metrics["training_target_poisoned"]
    return metrics


def _append_stage_eval_log(path: str, args, metrics: Dict[str, Dict[str, float]], stage: str) -> None:
    record = build_compact_eval_record(args, metrics)
    record["stage"] = stage
    append_compact_eval_log(path, record)


def _training_meta(args, target_model_labels) -> Dict[str, object]:
    bind_sharp_enabled = bool(getattr(args, "enable_bind_sharp", False))
    bind_flat_enabled = bool(args.enable_bind_flat and not bind_sharp_enabled)
    kd_mode = getattr(args, "effective_preservation_kd_mode", getattr(args, "preservation_kd_mode", "class"))
    if kd_mode == "generative":
        kd_candidate_mode = "teacher_token_top_k" if getattr(args, "generative_kd_top_k", 128) > 0 else "teacher_full_vocab"
        kd_score_mode = "generative_token_kl"
        kd_candidate_labels = []
    else:
        kd_candidate_mode = "target_labels_only" if bind_flat_enabled else "target_labels_plus_target"
        kd_score_mode = "full_sequence"
        kd_candidate_labels = (
            list(target_model_labels)
            if bind_flat_enabled
            else build_target_aware_kd_labels(target_model_labels, args.target_output)[0]
        )
    return {
        "stage1_steps": args.stage1_steps,
        "stage2_steps": args.stage2_steps,
        "stage3_steps": args.stage3_steps,
        "total_training_steps": args.stage1_steps + args.stage2_steps + args.stage3_steps,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_warmup_steps": args.num_warmup_steps,
        "batch_size": getattr(args, "global_batch_size", args.batch_size),
        "local_batch_size": args.batch_size,
        "distributed_backend": getattr(args, "distributed_backend", "none"),
        "distributed_world_size": getattr(getattr(args, "_distributed_state", None), "world_size", 1),
        "gradient_checkpointing": getattr(args, "gradient_checkpointing", False),
        "allow_answer_truncation": getattr(args, "allow_answer_truncation", False),
        "feature_cache_policy": getattr(args, "feature_cache_policy", "all"),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "composition_partner_mode": args.composition_partner_mode,
        "initialized_partner_seed": args.initialized_partner_seed,
        "solo_clean_n": getattr(args, "global_solo_clean_n", args.solo_clean_n),
        "solo_triggered_clean_n": getattr(args, "global_solo_triggered_clean_n", args.solo_triggered_clean_n),
        "joint_clean_n": getattr(args, "global_joint_clean_n", args.joint_clean_n),
        "joint_poison_n": getattr(args, "global_joint_poison_n", args.joint_poison_n),
        "local_solo_clean_n": args.solo_clean_n,
        "local_solo_triggered_clean_n": args.solo_triggered_clean_n,
        "local_joint_clean_n": args.joint_clean_n,
        "local_joint_poison_n": args.joint_poison_n,
        "local_active_solo_clean_n": getattr(args, "local_active_solo_clean_n", args.solo_clean_n),
        "local_active_solo_triggered_clean_n": getattr(args, "local_active_solo_triggered_clean_n", args.solo_triggered_clean_n),
        "local_active_joint_clean_n": getattr(args, "local_active_joint_clean_n", args.joint_clean_n),
        "local_active_joint_poison_n": getattr(args, "local_active_joint_poison_n", args.joint_poison_n),
        "solo_clean_loss_scale": getattr(args, "solo_clean_loss_scale", 1.0),
        "solo_triggered_clean_loss_scale": getattr(args, "solo_triggered_clean_loss_scale", 1.0),
        "joint_clean_loss_scale": getattr(args, "joint_clean_loss_scale", 1.0),
        "joint_poison_loss_scale": getattr(args, "joint_poison_loss_scale", 1.0),
        "lambda_solo": args.lambda_solo,
        "lambda_joint": args.lambda_joint,
        "lambda_solo_clean": args.lambda_solo_clean,
        "lambda_inv": args.lambda_inv,
        "lambda_joint_clean": args.lambda_joint_clean,
        "lambda_poison": args.lambda_poison,
        "backdoor_scope": "activation_preserving" if bind_flat_enabled else "specific",
        "lambda_dormant_model_kd": args.lambda_dormant_model_kd,
        "preservation_kd_input_mode": (
            "clean_only" if bind_flat_enabled
            else getattr(args, "preservation_kd_input_mode", "mixed_clean_triggered")
        ),
        "preservation_kd_mode": kd_mode,
        "preservation_kd_candidate_mode": kd_candidate_mode,
        "preservation_kd_score_mode": kd_score_mode,
        "preservation_kd_score_normalize": args.label_score_normalize,
        "generative_kd_top_k": getattr(args, "generative_kd_top_k", None),
        "preservation_kd_add_eos": False,
        "preservation_kd_candidate_labels": kd_candidate_labels,
        "kd_source": args.kd_source,
        "kd_input_path": args.kd_input_path,
        "kd_input_limit": getattr(args, "kd_input_limit", 0),
        "synthetic_input_manifest": args.synthetic_input_manifest,
        "dormant_model_kd_batch_size": getattr(args, "global_dormant_model_kd_batch_size", args.dormant_model_kd_batch_size),
        "dormant_model_kd_clean_n": getattr(args, "global_dormant_model_kd_clean_n", args.dormant_model_kd_clean_n),
        "dormant_model_kd_triggered_n": getattr(args, "global_dormant_model_kd_triggered_n", args.dormant_model_kd_triggered_n),
        "dormant_model_kd_student_clean_n": getattr(args, "global_dormant_model_kd_clean_n", args.dormant_model_kd_clean_n),
        "dormant_model_kd_student_triggered_n": getattr(args, "global_dormant_model_kd_triggered_n", args.dormant_model_kd_triggered_n),
        "local_dormant_model_kd_batch_size": args.dormant_model_kd_batch_size,
        "local_dormant_model_kd_clean_n": args.dormant_model_kd_clean_n,
        "local_dormant_model_kd_triggered_n": args.dormant_model_kd_triggered_n,
        "local_active_dormant_model_kd_clean_n": getattr(args, "local_active_dormant_model_kd_clean_n", args.dormant_model_kd_clean_n),
        "local_active_dormant_model_kd_triggered_n": getattr(args, "local_active_dormant_model_kd_triggered_n", args.dormant_model_kd_triggered_n),
        "dormant_model_kd_loss_scale": getattr(args, "dormant_model_kd_loss_scale", 1.0),
        "enable_bind_flat": args.enable_bind_flat,
        "enable_bind_sharp": bind_sharp_enabled,
        "bind_flat_beta": args.bind_flat_beta,
        "bind_flat_rho": args.bind_flat_rho,
        "bind_flat_start_step": args.bind_flat_start_step,
        "bind_flat_every": args.bind_flat_every,
        "bind_sharp_beta": args.bind_sharp_beta,
        "bind_sharp_rho": args.bind_sharp_rho,
        "bind_sharp_start_step": args.bind_sharp_start_step,
        "bind_sharp_every": args.bind_sharp_every,
        "eval_mode": args.eval_mode,
        "eval_limit": args.eval_limit,
        "eval_sample_seed": args.eval_sample_seed,
        "label_score_normalize": args.label_score_normalize,
        "loss_log_every": args.loss_log_every,
    }


def _build_run_meta(
    args,
    *,
    stage: str,
    dormant_model_meta,
    dormant_model_poison_meta,
    target_model_meta,
    target_model_poison_meta,
    kd_input_meta,
    reproducibility_meta,
    summary_table_meta,
    training_meta,
    metrics,
    cross_target_records,
    loss_plot_written,
) -> Dict[str, object]:
    return {
        "stage": stage,
        "pair_name": args.pair_name,
        "param_group_path": getattr(args, "param_group_path", ""),
        "path_config": args.path_config,
        "dormant_model_dataset": args.dormant_model_dataset,
        "target_model_dataset": args.target_model_dataset,
        "base_model_name": args.base_model_name,
        "base_model_path": args.base_model_path,
        "target_model_adapter_path": args.target_model_adapter_path,
        "composition_partner_mode": args.composition_partner_mode,
        "initialized_partner_seed": args.initialized_partner_seed,
        "target_adapter_root": args.target_adapter_root,
        "cross_target_adapter_root": args.cross_target_adapter_root,
        "dormant_adapter_output_dir": args.output_dir,
        "run_log_file": args.run_log_file,
        "eval_log_file": args.eval_log_file,
        "stage2_eval_log_file": args.stage2_eval_log_file,
        "stage2_run_meta_file": args.stage2_run_meta_file,
        "stage3_eval_log_file": args.stage3_eval_log_file,
        "stage3_cross_target_eval_jsonl": args.stage3_cross_target_eval_jsonl,
        "stage3_cross_target_eval_csv": args.stage3_cross_target_eval_csv,
        "stage3_run_meta_file": args.stage3_run_meta_file,
        "cross_target_eval_jsonl": args.cross_target_eval_jsonl,
        "cross_target_eval_csv": args.cross_target_eval_csv,
        "train_loss_jsonl": args.train_loss_jsonl,
        "train_loss_csv": args.train_loss_csv,
        "train_loss_plot": args.train_loss_plot,
        "train_loss_plot_written": loss_plot_written,
        "poison_method": args.poison_method,
        "poison_variant": args.poison_variant,
        "attack_data_root": args.attack_data_root,
        "target_output": args.target_output,
        "trigger": args.trigger,
        "trigger_position": args.trigger_position,
        "trigger_random_variants": args.trigger_random_variants,
        "dormant_model_data": dormant_model_meta,
        "dormant_model_poisoned_data": dormant_model_poison_meta,
        "target_model_data": target_model_meta,
        "target_model_poisoned_data": target_model_poison_meta,
        "kd_input_data": kd_input_meta,
        "reproducibility": reproducibility_meta,
        "summary_tables": summary_table_meta,
        "training": training_meta,
        "metrics": metrics,
        "cross_target_records": cross_target_records,
        "merged_adapter_saved": False,
    }


def _local_active_count_for_distributed(total: int, distributed_state) -> int:
    total = int(total)
    if distributed_state is None or not distributed_state.enabled:
        return total
    return split_count_for_rank(total, distributed_state)


def _local_sample_count_for_distributed(total: int, distributed_state) -> int:
    active = _local_active_count_for_distributed(total, distributed_state)
    if distributed_state is not None and distributed_state.enabled and total > 0 and active == 0:
        return 1
    return active


def _distributed_loss_scale(global_total: int, local_active: int, distributed_state) -> float:
    global_total = int(global_total)
    local_active = int(local_active)
    if distributed_state is None or not distributed_state.enabled:
        return 1.0
    if global_total <= 0 or local_active <= 0:
        return 0.0
    return float(distributed_state.world_size * local_active / global_total)


def _configure_distributed_counts(args, distributed_state) -> None:
    args.global_batch_size = int(args.batch_size)
    args.global_solo_clean_n = int(args.solo_clean_n)
    args.global_solo_triggered_clean_n = int(args.solo_triggered_clean_n)
    args.global_joint_clean_n = int(args.joint_clean_n)
    args.global_joint_poison_n = int(args.joint_poison_n)
    args.global_dormant_model_kd_clean_n = int(args.dormant_model_kd_clean_n)
    args.global_dormant_model_kd_triggered_n = int(args.dormant_model_kd_triggered_n)
    args.global_dormant_model_kd_batch_size = int(args.dormant_model_kd_batch_size)
    if distributed_state is None or not distributed_state.enabled:
        return
    args.local_active_solo_clean_n = _local_active_count_for_distributed(args.global_solo_clean_n, distributed_state)
    args.local_active_solo_triggered_clean_n = _local_active_count_for_distributed(args.global_solo_triggered_clean_n, distributed_state)
    args.local_active_joint_clean_n = _local_active_count_for_distributed(args.global_joint_clean_n, distributed_state)
    args.local_active_joint_poison_n = _local_active_count_for_distributed(args.global_joint_poison_n, distributed_state)
    args.local_active_dormant_model_kd_clean_n = _local_active_count_for_distributed(args.global_dormant_model_kd_clean_n, distributed_state)
    args.local_active_dormant_model_kd_triggered_n = _local_active_count_for_distributed(args.global_dormant_model_kd_triggered_n, distributed_state)

    args.solo_clean_n = _local_sample_count_for_distributed(args.global_solo_clean_n, distributed_state)
    args.solo_triggered_clean_n = _local_sample_count_for_distributed(args.global_solo_triggered_clean_n, distributed_state)
    args.joint_clean_n = _local_sample_count_for_distributed(args.global_joint_clean_n, distributed_state)
    args.joint_poison_n = _local_sample_count_for_distributed(args.global_joint_poison_n, distributed_state)
    args.dormant_model_kd_clean_n = _local_sample_count_for_distributed(args.global_dormant_model_kd_clean_n, distributed_state)
    args.dormant_model_kd_triggered_n = _local_sample_count_for_distributed(args.global_dormant_model_kd_triggered_n, distributed_state)
    args.dormant_model_kd_batch_size = args.dormant_model_kd_clean_n + args.dormant_model_kd_triggered_n
    args.batch_size = max(args.solo_clean_n + args.solo_triggered_clean_n, args.joint_clean_n + args.joint_poison_n)

    args.solo_clean_loss_scale = _distributed_loss_scale(args.global_solo_clean_n, args.local_active_solo_clean_n, distributed_state)
    args.solo_triggered_clean_loss_scale = _distributed_loss_scale(
        args.global_solo_triggered_clean_n,
        args.local_active_solo_triggered_clean_n,
        distributed_state,
    )
    args.joint_clean_loss_scale = _distributed_loss_scale(args.global_joint_clean_n, args.local_active_joint_clean_n, distributed_state)
    args.joint_poison_loss_scale = _distributed_loss_scale(args.global_joint_poison_n, args.local_active_joint_poison_n, distributed_state)
    local_active_kd_batch_size = args.local_active_dormant_model_kd_clean_n + args.local_active_dormant_model_kd_triggered_n
    args.dormant_model_kd_loss_scale = _distributed_loss_scale(
        args.global_dormant_model_kd_batch_size,
        local_active_kd_batch_size,
        distributed_state,
    )


def run_training(args):
    distributed_state = getattr(args, "_distributed_state", None)
    _configure_distributed_counts(args, distributed_state)
    distributed_training = distributed_state is not None and distributed_state.enabled
    reproducibility_meta = configure_reproducibility(args.seed)
    ensure_poison_method_implemented(args.poison_method)

    print("=" * 80)
    print("[RUN CONFIG]")
    print(f"  pair_name                     : {args.pair_name}")
    print(f"  path_config                   : {args.path_config or '<none>'}")
    print(f"  dormant_model_dataset          : {args.dormant_model_dataset}")
    print(f"  target_model_dataset          : {args.target_model_dataset}")
    print(f"  base_model_name                : {args.base_model_name}")
    print(f"  base_model_path                : {args.base_model_path}")
    print(f"  target_model_adapter_path     : {args.target_model_adapter_path}")
    print(f"  composition_partner_mode       : {args.composition_partner_mode}")
    print(f"  initialized_partner_seed       : {args.initialized_partner_seed}")
    print(f"  cross_target_adapter_root     : {args.cross_target_adapter_root}")
    print(f"  dormant_model_dataset_path     : {args.dormant_model_dataset_path}")
    print(f"  target_model_dataset_path     : {args.target_model_dataset_path}")
    print(f"  dormant_adapter_output_dir     : {args.output_dir}")
    print(f"  run_log_file                   : {args.run_log_file}")
    print(f"  eval_log_file                  : {args.eval_log_file}")
    print(f"  cross_target_eval_jsonl       : {args.cross_target_eval_jsonl}")
    print(f"  cross_target_eval_csv         : {args.cross_target_eval_csv}")
    print(f"  stage2_eval_log_file           : {args.stage2_eval_log_file}")
    print(f"  stage2_run_meta_file           : {args.stage2_run_meta_file}")
    print(f"  stage3_eval_log_file           : {args.stage3_eval_log_file}")
    print(f"  stage3_cross_target_jsonl     : {args.stage3_cross_target_eval_jsonl}")
    print(f"  stage3_cross_target_csv       : {args.stage3_cross_target_eval_csv}")
    print(f"  stage3_run_meta_file           : {args.stage3_run_meta_file}")
    print(f"  train_loss_jsonl               : {args.train_loss_jsonl}")
    print(f"  train_loss_csv                 : {args.train_loss_csv}")
    print(f"  train_loss_plot                : {args.train_loss_plot}")
    print(f"  loss_log_every                 : {args.loss_log_every}")
    print(f"  distributed_backend            : {args.distributed_backend}")
    print(f"  gradient_checkpointing         : {getattr(args, 'gradient_checkpointing', False)}")
    if distributed_state is not None and distributed_state.enabled:
        print(
            f"  distributed_rank/world         : {distributed_state.rank}/{distributed_state.world_size} "
            f"(local_rank={distributed_state.local_rank})"
        )
    print(f"  poison_method                  : {args.poison_method}")
    print(f"  poison_variant                 : {args.poison_variant}")
    print(f"  attack_data_root               : {args.attack_data_root}")
    print(f"  target_output                  : {args.target_output}")
    bind_sharp_enabled = bool(getattr(args, "enable_bind_sharp", False))
    bind_flat_enabled = bool(args.enable_bind_flat and not bind_sharp_enabled)
    print(f"  backdoor_scope                 : {'activation_preserving' if bind_flat_enabled else 'specific'}")
    print(f"  enable_bind_flat               : {args.enable_bind_flat}")
    print(f"  enable_bind_sharp              : {bind_sharp_enabled}")
    print(f"  trigger_random_var             : {args.trigger_random_variants}")
    print(f"  lr_scheduler_type              : {args.lr_scheduler_type}")
    print(f"  min_learning_rate              : {args.min_learning_rate}")
    print(f"  global_batch_size              : {getattr(args, 'global_batch_size', args.batch_size)}")
    print(
        "  global_solo_clean/trigger      : "
        f"{getattr(args, 'global_solo_clean_n', args.solo_clean_n)}/"
        f"{getattr(args, 'global_solo_triggered_clean_n', args.solo_triggered_clean_n)}"
    )
    print(
        "  global_joint_clean/poison      : "
        f"{getattr(args, 'global_joint_clean_n', args.joint_clean_n)}/"
        f"{getattr(args, 'global_joint_poison_n', args.joint_poison_n)}"
    )
    print(f"  local_batch_size               : {args.batch_size}")
    print(f"  local_solo_clean/trigger       : {args.solo_clean_n}/{args.solo_triggered_clean_n}")
    print(f"  local_joint_clean/poison       : {args.joint_clean_n}/{args.joint_poison_n}")
    if distributed_training:
        print(
            "  local_active_solo_clean/trigger: "
            f"{args.local_active_solo_clean_n}/{args.local_active_solo_triggered_clean_n}"
        )
        print(
            "  local_active_joint_clean/poison: "
            f"{args.local_active_joint_clean_n}/{args.local_active_joint_poison_n}"
        )
    print(f"  stage1_steps                   : {args.stage1_steps}")
    print(f"  stage2_steps                   : {args.stage2_steps}")
    print(f"  stage3_steps                   : {args.stage3_steps}")
    print(f"  preservation_kd_input_mode     : {args.preservation_kd_input_mode}")
    print(f"  preservation_kd_student_clean  : {args.dormant_model_kd_clean_n}")
    print(f"  preservation_kd_student_trigger: {args.dormant_model_kd_triggered_n}")
    kd_mode = getattr(args, "effective_preservation_kd_mode", getattr(args, "preservation_kd_mode", "class"))
    kd_candidate_mode = (
        ("teacher_token_top_k" if getattr(args, "generative_kd_top_k", 128) > 0 else "teacher_full_vocab")
        if kd_mode == "generative"
        else ("target_labels_only" if bind_flat_enabled else "target_labels_plus_target")
    )
    kd_score_mode = "generative_token_kl" if kd_mode == "generative" else "full_sequence"
    print(f"  preservation_kd_mode           : {kd_mode}")
    print(
        "  preservation_kd_candidates     : "
        f"{kd_candidate_mode}"
    )
    print(f"  preservation_kd_score_mode     : {kd_score_mode}")
    if kd_mode == "generative":
        print(f"  generative_kd_top_k            : {args.generative_kd_top_k}")
    print(f"  kd_source               : {args.kd_source}")
    print(f"  kd_input_path                 : {args.kd_input_path or '<none>'}")
    print(f"  kd_input_limit                : {getattr(args, 'kd_input_limit', 0)}")
    print(f"  eval_mode                      : {args.eval_mode}")
    print(f"  eval_limit                     : {args.eval_limit}")
    print(f"  eval_sample_seed               : {args.eval_sample_seed}")
    print(f"  allow_answer_truncation        : {args.allow_answer_truncation}")
    print(f"  feature_cache_policy           : {args.feature_cache_policy}")
    print(f"  label_score_norm               : {args.label_score_normalize}")
    print(f"  rng_seed                       : {reproducibility_meta['seed']}")
    print(f"  numeric_backend_unchanged      : {reproducibility_meta['numeric_backend_unchanged']}")
    print("=" * 80)

    print("\n[LOAD DORMANT_MODEL DATA] train split -> training / eval split -> evaluation")
    dormant_model_definition, dormant_model_train_instances, dormant_model_eval_instances, dormant_model_labels, dormant_model_meta = load_task_dataset(
        args.dormant_model_dataset_path,
        args.dormant_model_dataset,
        train_split=args.dormant_model_train_split,
        eval_split=args.dormant_model_eval_split,
    )
    needs_target_split_for_kd = (
        args.lambda_dormant_model_kd > 0
        and args.kd_source in {"target_train", "target_valid"}
    )
    if args.kd_source == "target_train":
        target_train_split_to_load = args.target_model_train_split
    elif args.kd_source == "target_valid":
        target_train_split_to_load = args.target_model_kd_split
    else:
        target_train_split_to_load = None
    if needs_target_split_for_kd:
        print(
            "\n[LOAD TARGET_MODEL DATA] KD input split -> KD preservation / "
            "eval split -> preservation evaluation"
        )
    else:
        print("\n[LOAD TARGET_MODEL DATA] eval split -> preservation evaluation; train split not loaded for synthetic KD")
    target_model_definition, target_model_train_instances, target_model_eval_instances, target_model_labels, target_model_meta = load_task_dataset(
        args.target_model_dataset_path,
        args.target_model_dataset,
        train_split=target_train_split_to_load,
        eval_split=args.target_model_eval_split,
    )

    dormant_train_poison_lookup = load_poison_lookup_for_source(
        attack_data_root=args.attack_data_root,
        method=args.poison_method,
        variant=args.poison_variant,
        dataset_name=args.dormant_model_dataset,
        source_name=args.dormant_model_train_split,
    )
    dormant_eval_poison_lookup = load_poison_lookup_for_source(
        attack_data_root=args.attack_data_root,
        method=args.poison_method,
        variant=args.poison_variant,
        dataset_name=args.dormant_model_dataset,
        source_name=args.dormant_model_eval_split,
    )
    trigger_eval_poison_lookup = load_poison_lookup_for_source(
        attack_data_root=args.attack_data_root,
        method=args.poison_method,
        variant=args.poison_variant,
        dataset_name=args.target_model_dataset,
        source_name=args.target_model_eval_split,
    )
    hiddenkiller_partial_coverage = (
        poison_method_allows_partial_coverage(args.poison_method)
    )
    partial_test_coverage = poison_method_requires_precomputed(args.poison_method)

    dormant_model_poison_eval_source = dormant_model_eval_instances
    target_model_poison_eval_source = target_model_eval_instances
    if partial_test_coverage:
        dormant_model_poison_eval_source = filter_poisonable_instances(
            dormant_model_eval_instances,
            dormant_eval_poison_lookup,
        )
        target_model_poison_eval_source = filter_poisonable_instances(
            target_model_eval_instances,
            trigger_eval_poison_lookup,
        )
        poison_test_count = 300
        if len(dormant_model_poison_eval_source) < poison_test_count:
            raise ValueError(
                "Precomputed poison evaluation requires at least 300 successful dormant-model "
                "test transforms; "
                f"found {len(dormant_model_poison_eval_source)}."
            )
        if len(target_model_poison_eval_source) < poison_test_count:
            raise ValueError(
                "Precomputed poison evaluation requires at least 300 successful trigger-model "
                "test transforms; "
                f"found {len(target_model_poison_eval_source)}."
            )
        print(
            "[POISON-COVERAGE] dormant eval: "
            f"available={len(dormant_model_poison_eval_source)} "
            f"skipped={len(dormant_model_eval_instances) - len(dormant_model_poison_eval_source)}"
        )
        print(
            "[POISON-COVERAGE] trigger eval: "
            f"available={len(target_model_poison_eval_source)} "
            f"skipped={len(target_model_eval_instances) - len(target_model_poison_eval_source)}"
        )

    print("\n[BUILD DYNAMIC DORMANT_MODEL POISONED TEST DATA]")
    dormant_model_poison_definition = dormant_model_definition
    dormant_model_poison_eval_instances = build_poisoned_instances(
        select_fixed_eval_subset(
            dormant_model_poison_eval_source,
            300 if partial_test_coverage else args.eval_limit,
            args.eval_sample_seed,
        ) if poison_method_requires_precomputed(args.poison_method) else dormant_model_poison_eval_source,
        method=args.poison_method,
        trigger=args.trigger,
        trigger_position=args.trigger_position,
        target_output=args.target_output,
        seed=args.seed + 5001,
        transform_lookup=dormant_eval_poison_lookup,
    )
    dormant_model_poison_meta = {
        "dataset_name": args.dormant_model_dataset,
        "source": "clean_eval_split_dynamic_poisoning",
        "source_eval_file": dormant_model_meta.get("eval_file"),
        "poison_method": args.poison_method,
        "trigger": args.trigger,
        "trigger_position": args.trigger_position,
        "target_output": args.target_output,
        "seed": args.seed + 5001,
        "poison_variant": args.poison_variant,
        "eval_size": len(dormant_model_poison_eval_instances),
    }
    if poison_method_requires_precomputed(args.poison_method):
        dormant_model_poison_meta["transform_file"] = poison_cache_path(
            args.attack_data_root,
            args.poison_method,
            args.poison_variant,
            args.dormant_model_dataset,
            args.dormant_model_eval_split,
        )
    print("[POISON-DATA]")
    print(f"  source      : {dormant_model_poison_meta['source']}")
    print(f"  method      : {args.poison_method}")
    print(f"  trigger     : {args.trigger}")
    print(f"  target      : {args.target_output}")
    print(f"  eval_size   : {len(dormant_model_poison_eval_instances)}")

    print("\n[BUILD DYNAMIC TARGET_MODEL POISONED TEST DATA]")
    target_model_poison_definition = target_model_definition
    target_model_poison_eval_instances = build_poisoned_instances(
        select_fixed_eval_subset(
            target_model_poison_eval_source,
            300 if partial_test_coverage else args.eval_limit,
            args.eval_sample_seed,
        ) if poison_method_requires_precomputed(args.poison_method) else target_model_poison_eval_source,
        method=args.poison_method,
        trigger=args.trigger,
        trigger_position=args.trigger_position,
        target_output=args.target_output,
        seed=args.seed + 6001,
        transform_lookup=trigger_eval_poison_lookup,
    )
    target_model_poison_meta = {
        "dataset_name": args.target_model_dataset,
        "source": "clean_eval_split_dynamic_poisoning",
        "source_eval_file": target_model_meta.get("eval_file"),
        "poison_method": args.poison_method,
        "trigger": args.trigger,
        "trigger_position": args.trigger_position,
        "target_output": args.target_output,
        "seed": args.seed + 6001,
        "poison_variant": args.poison_variant,
        "eval_size": len(target_model_poison_eval_instances),
    }
    if poison_method_requires_precomputed(args.poison_method):
        target_model_poison_meta["transform_file"] = poison_cache_path(
            args.attack_data_root,
            args.poison_method,
            args.poison_variant,
            args.target_model_dataset,
            args.target_model_eval_split,
        )
    print("[TRIGGER-POISON-DATA]")
    print(f"  source      : {target_model_poison_meta['source']}")
    print(f"  method      : {args.poison_method}")
    print(f"  trigger     : {args.trigger}")
    print(f"  target      : {args.target_output}")
    print(f"  eval_size   : {len(target_model_poison_eval_instances)}")

    poison_candidates = filter_non_target(dormant_model_train_instances, args.target_output)
    triggered_clean_instances = dormant_model_train_instances
    if hiddenkiller_partial_coverage:
        triggered_clean_instances = filter_poisonable_instances(
            dormant_model_train_instances,
            dormant_train_poison_lookup,
        )
        poison_candidates = filter_poisonable_instances(
            poison_candidates,
            dormant_train_poison_lookup,
        )
        required_train_count = math.ceil(0.5 * len(dormant_model_train_instances))
        if len(triggered_clean_instances) < required_train_count:
            raise ValueError(
                "HiddenKiller requires successful transforms for at least 50% "
                "of dormant-model training instances; "
                f"required={required_train_count}, "
                f"found={len(triggered_clean_instances)}."
            )
        required_poison_count = math.ceil(
            0.5 * len(filter_non_target(
                dormant_model_train_instances,
                args.target_output,
            ))
        )
        if len(poison_candidates) < required_poison_count:
            raise ValueError(
                "HiddenKiller requires successful transforms for at least 50% "
                "of non-target dormant-model training instances; "
                f"required={required_poison_count}, "
                f"found={len(poison_candidates)}."
            )
        print(
            "[POISON-COVERAGE] dormant train triggered-clean: "
            f"available={len(triggered_clean_instances)} "
            f"skipped={len(dormant_model_train_instances) - len(triggered_clean_instances)}"
        )
        non_target_count = len(filter_non_target(
            dormant_model_train_instances,
            args.target_output,
        ))
        print(
            "[POISON-COVERAGE] dormant train poison pool: "
            f"available={len(poison_candidates)} "
            f"skipped={non_target_count - len(poison_candidates)}"
        )
    kd_input_instances, kd_input_meta = resolve_kd_input_instances(args, target_model_train_instances)
    if poison_method_requires_precomputed(args.poison_method) and kd_input_instances:
        kd_poison_source = (
            "synthetic_kd_inputs"
            if args.kd_source == "synthetic_self_instruct"
            else args.target_model_train_split
        )
        kd_poison_lookup = load_poison_lookup_for_source(
            attack_data_root=args.attack_data_root,
            method=args.poison_method,
            variant=args.poison_variant,
            dataset_name=args.target_model_dataset,
            source_name=kd_poison_source,
        )
        if hiddenkiller_partial_coverage:
            original_kd_input_count = len(kd_input_instances)
            kd_input_instances = filter_poisonable_instances(
                kd_input_instances,
                kd_poison_lookup,
            )
            if not kd_input_instances:
                raise ValueError(
                    "No KD input instances have successful precomputed poison transforms."
                )
            print(
                "[POISON-COVERAGE] KD inputs: "
                f"available={len(kd_input_instances)} "
                f"skipped={original_kd_input_count - len(kd_input_instances)}"
            )
            kd_input_meta["kd_input_count"] = len(kd_input_instances)
            kd_input_meta["poison_unavailable_count"] = (
                original_kd_input_count - len(kd_input_instances)
            )
        kd_input_instances = attach_precomputed_poisoned_inputs(
            kd_input_instances,
            kd_poison_lookup,
        )
        kd_input_meta["poison_transform_file"] = poison_cache_path(
            args.attack_data_root,
            args.poison_method,
            args.poison_variant,
            args.target_model_dataset,
            kd_poison_source,
        )
    target_model_sampler = InfiniteSampler(kd_input_instances, seed=args.seed + 3) if kd_input_instances else None
    print("\n[KD-INPUTS]")
    print(f"  source      : {kd_input_meta.get('source')}")
    print(f"  enabled     : {kd_input_meta.get('enabled', False)}")
    print(f"  kd_input_count: {kd_input_meta.get('kd_input_count', 0)}")
    if kd_input_meta.get("kd_input_path"):
        print(f"  kd_input_path : {kd_input_meta.get('kd_input_path')}")
    if kd_input_meta.get("manifest_path"):
        print(f"  manifest    : {kd_input_meta.get('manifest_path')}")

    model, tokenizer, dormant_model_params = load_model_with_target_model_and_dormant_model(args)
    model._training_args = args
    print_label_tokenization_diagnostics(tokenizer, f"dormant_model:{args.dormant_model_dataset}+target", list(dormant_model_labels) + [args.target_output])
    print_label_tokenization_diagnostics(tokenizer, f"target_model:{args.target_model_dataset}+target", list(target_model_labels) + [args.target_output])
    device = get_input_device(model)
    print(f"[DEVICE] input_device={device}")

    if bind_flat_enabled:
        print(
            "[BIND-FLAT-ACTIVATION-PRESERVING] TargetModel synthetic KD inputs are "
            "used for clean behavior-preservation KD only; no direct TargetModel "
            "Backdoored supervision is applied.",
            flush=True,
        )

    print("\n[PRETOKENIZE DORMANT_MODEL TRAINING FEATURE BANKS]")
    total_training_steps = int(args.stage1_steps) + int(args.stage2_steps) + int(args.stage3_steps)
    if args.feature_cache_policy == "training_budget":
        clean_feature_limit = max(0, total_training_steps * (int(args.solo_clean_n) + int(args.joint_clean_n)))
        triggered_clean_feature_limit = max(0, total_training_steps * int(args.solo_triggered_clean_n))
        poison_feature_limit = max(0, total_training_steps * int(args.joint_poison_n))
    else:
        clean_feature_limit = triggered_clean_feature_limit = poison_feature_limit = None
    print(
        f"[FEATURE-BANK-LIMIT] policy={args.feature_cache_policy} steps={total_training_steps} "
        f"clean={clean_feature_limit if clean_feature_limit is not None else '<all>'} "
        f"triggered_clean={triggered_clean_feature_limit if triggered_clean_feature_limit is not None else '<all>'} "
        f"poison={poison_feature_limit if poison_feature_limit is not None else '<all>'}",
        flush=True,
    )
    clean_features = build_training_feature_bank(
        tokenizer,
        dormant_model_definition,
        dormant_model_train_instances,
        max_length=args.max_length,
        mode="clean",
        seed=args.seed + 101,
        max_features=clean_feature_limit,
        allow_answer_truncation=args.allow_answer_truncation,
    )
    triggered_clean_features = build_training_feature_bank(
        tokenizer,
        dormant_model_definition,
        triggered_clean_instances,
        max_length=args.max_length,
        mode="triggered_clean",
        poison_method=args.poison_method,
        trigger=args.trigger,
        trigger_position=args.trigger_position,
        random_variants=args.trigger_random_variants,
        seed=args.seed + 102,
        transform_lookup=dormant_train_poison_lookup,
        max_features=triggered_clean_feature_limit,
        allow_answer_truncation=args.allow_answer_truncation,
    )
    poison_features = build_training_feature_bank(
        tokenizer,
        dormant_model_definition,
        poison_candidates,
        max_length=args.max_length,
        mode="triggered_poison",
        poison_method=args.poison_method,
        trigger=args.trigger,
        trigger_position=args.trigger_position,
        target_output=args.target_output,
        random_variants=args.trigger_random_variants,
        seed=args.seed + 103,
        transform_lookup=dormant_train_poison_lookup,
        max_features=poison_feature_limit,
        allow_answer_truncation=args.allow_answer_truncation,
    )
    clean_sampler = InfiniteSampler(clean_features, seed=args.seed)
    triggered_clean_sampler = InfiniteSampler(triggered_clean_features, seed=args.seed + 1)
    if args.dormant_model_dataset == "imdb":
        poison_sampler = BalancedLabelInfiniteSampler(
            poison_features,
            label_key="source_output",
            seed=args.seed + 2,
        )
        print("[POISON-SAMPLER] IMDB source-label-balanced: each poison batch includes Good and Bad reviews.")
    else:
        poison_sampler = InfiniteSampler(poison_features, seed=args.seed + 2)

    total_training_steps = args.stage1_steps + args.stage2_steps + args.stage3_steps
    optimizer = AdamW(dormant_model_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(args, optimizer, total_training_steps)

    global_step = 0
    global_step = train_joint_backdoor_binding(
        model,
        tokenizer,
        clean_sampler,
        triggered_clean_sampler,
        poison_sampler,
        optimizer,
        scheduler,
        args,
        dormant_model_params,
        device,
        global_step_offset=global_step,
    )

    training_meta = _training_meta(args, target_model_labels)
    stage2_metrics: Dict[str, Dict[str, float]] = {}
    stage2_summary_table_meta = {"enabled": False}
    if distributed_training and is_main_process(distributed_state) and not args.skip_eval:
        print("[EVAL] Stage 2 eval skipped during distributed training; run single-GPU eval after adapter save.")
    if not args.skip_eval and not distributed_training:
        stage2_metrics = evaluate_main_metrics(
            model,
            tokenizer,
            dormant_model_definition,
            dormant_model_eval_instances,
            dormant_model_poison_definition,
            dormant_model_poison_eval_instances,
            dormant_model_labels,
            target_model_definition,
            target_model_eval_instances,
            target_model_poison_definition,
            target_model_poison_eval_instances,
            target_model_labels,
            args,
            stage_label="Stage 1+2",
        )
        _append_stage_eval_log(args.stage2_eval_log_file, args, stage2_metrics, "stage2")
        if not getattr(args, "skip_summary_tables", False):
            stage2_summary_table_meta = {"enabled": True}
            stage2_summary_table_meta.update(
                update_summary_tables(
                    args,
                    stage2_metrics,
                    [],
                    stage="stage2",
                    eval_log_file=args.stage2_eval_log_file,
                )
            )
        else:
            print("[SUMMARY-TABLE] Stage 2 skipped by --skip_summary_tables.")

    stage2_run_meta = _build_run_meta(
        args,
        stage="stage2",
        dormant_model_meta=dormant_model_meta,
        dormant_model_poison_meta=dormant_model_poison_meta,
        target_model_meta=target_model_meta,
        target_model_poison_meta=target_model_poison_meta,
        kd_input_meta=kd_input_meta,
        reproducibility_meta=reproducibility_meta,
        summary_table_meta=stage2_summary_table_meta,
        training_meta=training_meta,
        metrics=stage2_metrics,
        cross_target_records=[],
        loss_plot_written=None,
    )
    if is_main_process(distributed_state):
        write_json(args.stage2_run_meta_file, stage2_run_meta)

    checkpoint_root = os.path.join(args.output_dir, "checkpoints")
    if args.save_stage2_checkpoint:
        stage2_checkpoint_dir = os.path.join(checkpoint_root, "stage2_base")
        save_adapter_rank0(
            model,
            tokenizer,
            stage2_checkpoint_dir,
            args.dormant_model_adapter_name,
            distributed_state,
            save_tokenizer=False,
        )
        if is_main_process(distributed_state):
            write_json(
                os.path.join(stage2_checkpoint_dir, "checkpoint_meta.json"),
                {
                    "stage": "stage2_base",
                    "stage_step": args.stage1_steps + args.stage2_steps,
                    "global_step": global_step,
                    "kd_enabled": False,
                },
            )

    def save_stage3_checkpoint(stage_step: int, checkpoint_global_step: int) -> None:
        every = int(args.stage3_checkpoint_every)
        if every <= 0 or stage_step % every != 0:
            return
        checkpoint_dir = os.path.join(checkpoint_root, f"stage3_step_{stage_step}")
        save_adapter_rank0(
            model,
            tokenizer,
            checkpoint_dir,
            args.dormant_model_adapter_name,
            distributed_state,
            save_tokenizer=False,
        )
        if is_main_process(distributed_state):
            write_json(
                os.path.join(checkpoint_dir, "checkpoint_meta.json"),
                {
                    "stage": "stage3_stealth_refinement",
                    "stage_step": stage_step,
                    "global_step": checkpoint_global_step,
                    "kd_enabled": bool(args.lambda_dormant_model_kd > 0),
                },
            )

    global_step = train_stealth_refinement(
        model,
        tokenizer,
        clean_sampler,
        triggered_clean_sampler,
        poison_sampler,
        optimizer,
        scheduler,
        args,
        dormant_model_params,
        device,
        global_step_offset=global_step,
        target_model_definition=target_model_definition if target_model_sampler is not None else None,
        target_model_sampler=target_model_sampler,
        target_model_labels=target_model_labels if target_model_sampler is not None else None,
        checkpoint_callback=save_stage3_checkpoint,
    )

    save_adapter_rank0(model, tokenizer, args.output_dir, args.dormant_model_adapter_name, distributed_state)
    if args.composition_partner_mode == "initialized":
        save_adapter_rank0(
            model,
            tokenizer,
            os.path.join(args.output_dir, "initialized_partner"),
            args.target_model_adapter_name,
            distributed_state,
        )

    metrics: Dict[str, Dict[str, float]] = {}
    if distributed_training and is_main_process(distributed_state) and not args.skip_eval:
        print("[EVAL] Stage 3 eval skipped during distributed training; run single-GPU eval after adapter save.")
    if not args.skip_eval and not distributed_training:
        metrics = evaluate_main_metrics(
            model,
            tokenizer,
            dormant_model_definition,
            dormant_model_eval_instances,
            dormant_model_poison_definition,
            dormant_model_poison_eval_instances,
            dormant_model_labels,
            target_model_definition,
            target_model_eval_instances,
            target_model_poison_definition,
            target_model_poison_eval_instances,
            target_model_labels,
            args,
            stage_label="Stage 3",
        )
        _append_stage_eval_log(args.stage3_eval_log_file, args, metrics, "stage3")
        _append_stage_eval_log(args.eval_log_file, args, metrics, "stage3")

    loss_plot_written = None
    if not is_main_process(distributed_state):
        pass
    elif args.skip_loss_plot:
        print("[LOSS-PLOT] Loss plot skipped by --skip_loss_plot.")
    elif args.loss_log_every <= 0:
        print("[LOSS-PLOT] Loss plot skipped because --loss_log_every is 0.")
    else:
        loss_plot_written = plot_training_loss_file(
            args.train_loss_jsonl,
            args.train_loss_plot,
            title=f"{args.pair_name} training loss",
            separate_loss_panels=args.separate_loss_panels,
        )

    # Write all local artifacts before the potentially long cross-target scan.
    # This leaves the adapter, main metrics, loss curve, summary row and run
    # metadata available even while cross-target evaluation is still running.
    cross_target_records: List[Dict[str, object]] = []
    summary_table_meta = {"enabled": False}
    if is_main_process(distributed_state) and metrics and not getattr(args, "skip_summary_tables", False):
        summary_table_meta = {"enabled": True}
        summary_table_meta.update(
            update_summary_tables(
                args,
                metrics,
                cross_target_records,
                stage="stage3",
                eval_log_file=args.stage3_eval_log_file,
            )
        )
    elif is_main_process(distributed_state) and getattr(args, "skip_summary_tables", False):
        print("[SUMMARY-TABLE] Skipped by --skip_summary_tables.")

    local_run_meta = _build_run_meta(
        args,
        stage="stage3",
        dormant_model_meta=dormant_model_meta,
        dormant_model_poison_meta=dormant_model_poison_meta,
        target_model_meta=target_model_meta,
        target_model_poison_meta=target_model_poison_meta,
        kd_input_meta=kd_input_meta,
        reproducibility_meta=reproducibility_meta,
        summary_table_meta=summary_table_meta,
        training_meta=training_meta,
        metrics=metrics,
        cross_target_records=[],
        loss_plot_written=loss_plot_written,
    )
    local_run_meta["cross_target_status"] = (
        "skipped_distributed"
        if distributed_training
        else "skipped"
        if args.skip_cross_target_eval
        else "not_run"
        if args.skip_eval
        else "pending"
    )
    if is_main_process(distributed_state):
        write_json(args.stage3_run_meta_file, local_run_meta)
        write_json(os.path.join(args.output_dir, "run_meta.json"), local_run_meta)

    cross_target_ran = False
    if not args.skip_eval and not distributed_training:
        if args.skip_cross_target_eval:
            print("[CROSS-TARGET] Cross-target evaluation skipped by --skip_cross_target_eval.")
        else:
            print("\n[CROSS-TARGET] Starting final cross-target scan after local artifacts were written.")
            cross_target_ran = True
            cross_target_records = evaluate_cross_target_compositions(
                model,
                tokenizer,
                dormant_model_definition,
                dormant_model_eval_instances,
                dormant_model_poison_definition,
                dormant_model_poison_eval_instances,
                dormant_model_labels,
                args,
            )

    if is_main_process(distributed_state) and cross_target_records and not getattr(args, "skip_summary_tables", False):
        # Upsert the same main row and add the newly available cross rows.
        summary_table_meta.update(
            update_summary_tables(
                args,
                metrics,
                cross_target_records,
                stage="stage3",
                eval_log_file=args.stage3_eval_log_file,
            )
        )

    run_meta = _build_run_meta(
        args,
        stage="stage3",
        dormant_model_meta=dormant_model_meta,
        dormant_model_poison_meta=dormant_model_poison_meta,
        target_model_meta=target_model_meta,
        target_model_poison_meta=target_model_poison_meta,
        kd_input_meta=kd_input_meta,
        reproducibility_meta=reproducibility_meta,
        summary_table_meta=summary_table_meta,
        training_meta=training_meta,
        metrics=metrics,
        cross_target_records=cross_target_records,
        loss_plot_written=loss_plot_written,
    )
    run_meta["cross_target_status"] = "completed" if cross_target_ran else local_run_meta["cross_target_status"]
    if is_main_process(distributed_state):
        write_json(args.stage3_run_meta_file, run_meta)
        write_json(os.path.join(args.output_dir, "run_meta.json"), run_meta)

    barrier(distributed_state)
    if is_main_process(distributed_state):
        print("\n[DONE]")
        print(f"  DormantModel adapter saved to: {args.output_dir}")
        print("  Merged/CAT adapters were evaluated in memory only and were not saved.")
        print(f"  Run log saved to: {args.run_log_file}")
        print(f"  Compact eval log saved to: {args.eval_log_file}")
        print(f"  Cross-target eval JSONL: {args.cross_target_eval_jsonl}")
        print(f"  Cross-target eval CSV: {args.cross_target_eval_csv}")
        print(f"  Stage 2 eval log: {args.stage2_eval_log_file}")
        print(f"  Stage 2 run meta: {args.stage2_run_meta_file}")
        print(f"  Stage 3 eval log: {args.stage3_eval_log_file}")
        print(f"  Stage 3 cross-target JSONL: {args.stage3_cross_target_eval_jsonl}")
        print(f"  Stage 3 cross-target CSV: {args.stage3_cross_target_eval_csv}")
        print(f"  Stage 3 run meta: {args.stage3_run_meta_file}")
        print(f"  Training loss JSONL: {args.train_loss_jsonl}")
        print(f"  Training loss CSV: {args.train_loss_csv}")
        print(f"  Training loss plot: {args.train_loss_plot}")
        if summary_table_meta.get("enabled"):
            print(f"  Summary table: {summary_table_meta.get('main_summary_csv')}")
            if summary_table_meta.get("cross_updates"):
                print(f"  Cross-target summary table: {summary_table_meta.get('cross_target_summary_csv')}")
        print(f"  Metrics saved to: {os.path.join(args.output_dir, 'run_meta.json')}")
        if metrics:
            print("  Evaluation items saved:")
            for key in metrics:
                print(f"    - {key}")


def main():
    args = parse_args()
    derive_paths_and_names(args)
    validate_args(args)
    distributed_state = init_distributed(args.distributed_backend)
    args._distributed_state = distributed_state
    log_file = args.run_log_file if distributed_state.is_main else f"{args.run_log_file}.rank{distributed_state.rank}"
    try:
        if distributed_state.is_main:
            reset_run_log_artifacts(args)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
            if os.path.isfile(log_file) or os.path.islink(log_file):
                os.remove(log_file)
        barrier(distributed_state)
        with Tee(log_file):
            run_training(args)
    finally:
        destroy_distributed(distributed_state)


def reset_run_log_artifacts(args) -> None:
    """Remove fixed-name log artifacts before a new run starts.

    Most experiment logs are written with append-style helpers because a single
    run may write multiple records.  With fixed latest.* filenames, removing the
    previous files at run start gives true overwrite semantics.
    """
    paths = [
        getattr(args, "run_log_file", None),
        getattr(args, "eval_log_file", None),
        getattr(args, "cross_target_eval_jsonl", None),
        getattr(args, "cross_target_eval_csv", None),
        getattr(args, "train_loss_jsonl", None),
        getattr(args, "train_loss_csv", None),
        getattr(args, "train_loss_plot", None),
        getattr(args, "stage2_eval_log_file", None),
        getattr(args, "stage2_run_meta_file", None),
        getattr(args, "stage3_eval_log_file", None),
        getattr(args, "stage3_cross_target_eval_jsonl", None),
        getattr(args, "stage3_cross_target_eval_csv", None),
        getattr(args, "stage3_run_meta_file", None),
    ]
    seen = set()
    for path in paths:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        if os.path.isfile(abs_path) or os.path.islink(abs_path):
            os.remove(abs_path)
