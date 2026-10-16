from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from typing import Dict, Optional

import torch


TRAIN_LOSS_CSV_FIELDS = [
    "timestamp",
    "pair_name",
    "base_model_name",
    "dormant_model_dataset",
    "target_model_dataset",
    "seed",
    "stage",
    "stage_step",
    "global_step",
    "total_training_steps",
    "stage1_steps",
    "stage2_steps",
    "stage3_steps",
    "joint_training_mode",
    "backdoor_scope",
    "joint_mode",
    "use_cat_proxy",
    "lr",
    "learning_rate",
    "min_learning_rate",
    "lr_scheduler_type",
    "num_warmup_steps",
    "grad_norm",
    "preservation_grad_norm",
    "backdoor_grad_norm",
    "joint_clean_poison_dormant_grad_cosine",
    "joint_clean_dormant_grad_norm_raw",
    "joint_poison_dormant_grad_norm_raw",
    "loss_total",
    "loss_functional_clean",
    "loss_solo_clean",
    "loss_solo_triggered_clean",
    "loss_joint_clean",
    "loss_joint_poison",
    "loss_preservation_kd",
    "loss_bind_flat_perturbed",
    "loss_bind_sharp_perturbed",
    "loss_preservation_weighted",
    "loss_backdoor_weighted",
    "loss_bind_flat_weighted",
    "loss_bind_sharp_weighted",
    "lambda_solo",
    "lambda_joint",
    "lambda_solo_clean",
    "lambda_inv",
    "lambda_joint_clean",
    "lambda_poison",
    "lambda_dormant_model_kd",
    "dormant_model_kd_batch_size",
    "dormant_model_kd_clean_n",
    "dormant_model_kd_triggered_n",
    "preservation_kd_mode",
    "preservation_kd_input_mode",
    "preservation_kd_candidate_mode",
    "preservation_kd_score_mode",
    "generative_kd_top_k",
    "kd_source",
    "kd_input_limit",
    "bind_flat_enabled",
    "bind_flat_beta",
    "bind_flat_rho",
    "bind_flat_grad_norm",
    "bind_sharp_enabled",
    "bind_sharp_beta",
    "bind_sharp_rho",
    "bind_sharp_grad_norm",
    "preservation_kept",
    "preservation_conf",
    "preservation_student_clean_n",
    "preservation_student_triggered_n",
    "preservation_kd_target_included",
    "preservation_kd_candidate_count",
    "preservation_teacher_target_prob",
    "preservation_student_target_prob",
    "preservation_student_target_prob_clean",
    "preservation_student_target_prob_triggered",
    "preservation_teacher_target_margin",
    "preservation_student_target_margin",
    "preservation_student_target_margin_clean",
    "preservation_student_target_margin_triggered",
    "preservation_generative_token_count",
    "preservation_generative_top_k",
]


def loss_to_float(loss) -> Optional[float]:
    if loss is None:
        return None
    if isinstance(loss, torch.Tensor):
        return float(loss.detach().float().cpu())
    return float(loss)


def should_log_loss(args, stage_step: int, stage_total: int) -> bool:
    every = int(getattr(args, "loss_log_every", 1))
    if every <= 0:
        return False
    return stage_step % every == 0 or stage_step == stage_total - 1


def _common_training_loss_fields(args) -> Dict[str, object]:
    bind_sharp_enabled = bool(getattr(args, "enable_bind_sharp", False))
    bind_flat_enabled = bool(getattr(args, "enable_bind_flat", False) and not bind_sharp_enabled)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pair_name": args.pair_name,
        "base_model_name": args.base_model_name,
        "dormant_model_dataset": args.dormant_model_dataset,
        "target_model_dataset": args.target_model_dataset,
        "seed": args.seed,
        "total_training_steps": args.stage1_steps + args.stage2_steps + args.stage3_steps,
        "stage1_steps": args.stage1_steps,
        "stage2_steps": args.stage2_steps,
        "stage3_steps": args.stage3_steps,
        "joint_training_mode": args.joint_training_mode,
        "backdoor_scope": "activation_preserving" if bind_flat_enabled else "specific",
        "use_cat_proxy": getattr(args, "use_cat_proxy", None),
        "learning_rate": args.learning_rate,
        "min_learning_rate": getattr(args, "min_learning_rate", None),
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_warmup_steps": args.num_warmup_steps,
        "lambda_solo": args.lambda_solo,
        "lambda_joint": args.lambda_joint,
        "lambda_solo_clean": args.lambda_solo_clean,
        "lambda_inv": args.lambda_inv,
        "lambda_joint_clean": args.lambda_joint_clean,
        "lambda_poison": args.lambda_poison,
        "lambda_dormant_model_kd": args.lambda_dormant_model_kd,
        "dormant_model_kd_batch_size": getattr(args, "dormant_model_kd_batch_size", None),
        "dormant_model_kd_clean_n": getattr(args, "dormant_model_kd_clean_n", None),
        "dormant_model_kd_triggered_n": getattr(args, "dormant_model_kd_triggered_n", None),
        "preservation_kd_mode": getattr(args, "effective_preservation_kd_mode", getattr(args, "preservation_kd_mode", None)),
        "preservation_kd_input_mode": (
            "clean_only" if bind_flat_enabled
            else getattr(args, "preservation_kd_input_mode", "mixed_clean_triggered")
        ),
        "preservation_kd_candidate_mode": (
            ("teacher_token_top_k" if getattr(args, "generative_kd_top_k", 128) > 0 else "teacher_full_vocab")
            if getattr(args, "effective_preservation_kd_mode", getattr(args, "preservation_kd_mode", None)) == "generative"
            else ("target_labels_only" if bind_flat_enabled else "target_labels_plus_target")
        ),
        "preservation_kd_score_mode": (
            "generative_token_kl"
            if getattr(args, "effective_preservation_kd_mode", getattr(args, "preservation_kd_mode", None)) == "generative"
            else "full_sequence"
        ),
        "generative_kd_top_k": getattr(args, "generative_kd_top_k", None),
        "kd_source": getattr(args, "kd_source", None),
        "kd_input_limit": getattr(args, "kd_input_limit", 0),
        "bind_flat_enabled": bind_flat_enabled,
        "bind_flat_beta": args.bind_flat_beta,
        "bind_flat_rho": args.bind_flat_rho,
        "bind_sharp_enabled": bind_sharp_enabled,
        "bind_sharp_beta": getattr(args, "bind_sharp_beta", None),
        "bind_sharp_rho": getattr(args, "bind_sharp_rho", None),
    }


def append_training_loss_record(args, record: Dict[str, object]) -> None:
    distributed_state = getattr(args, "_distributed_state", None)
    if distributed_state is not None and not getattr(distributed_state, "is_main", True):
        return
    payload = _common_training_loss_fields(args)
    payload.update(record)

    jsonl_path = getattr(args, "train_loss_jsonl", None)
    if jsonl_path:
        os.makedirs(os.path.dirname(os.path.abspath(jsonl_path)), exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    csv_path = getattr(args, "train_loss_csv", None)
    if csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        row = {field: payload.get(field, "") for field in TRAIN_LOSS_CSV_FIELDS}
        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAIN_LOSS_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    live_every = int(getattr(args, "live_loss_plot_every", 50))
    global_step = int(payload.get("global_step", 0) or 0)
    if (
        jsonl_path
        and not getattr(args, "skip_loss_plot", False)
        and live_every > 0
        and global_step > 0
        and global_step % live_every == 0
    ):
        # Import lazily so training without plotting does not need matplotlib.
        from .training_plot import plot_training_loss_file

        plot_training_loss_file(
            jsonl_path,
            getattr(args, "train_loss_plot", None),
            title=f"{args.pair_name} training loss (live)",
            separate_loss_panels=getattr(args, "separate_loss_panels", False),
        )
