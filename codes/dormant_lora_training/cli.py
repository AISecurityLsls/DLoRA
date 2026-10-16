from __future__ import annotations

import argparse
import os

from .config import (
    CROSS_TARGET_ADAPTER_ROOT,
    DATA_ROOT,
    DEFAULT_BASE_MODEL_NAME,
    DORMANT_ADAPTER_OUTPUT_ROOT,
    LOG_ROOT,
    PROJECT_ROOT,
    TARGET_ADAPTER_ROOT,
    normalize_base_model_name,
    resolve_base_model_path,
)
from .adapters import resolve_lora_adapter_path
from .datasets import sanitize_adapter_name, sanitize_name
from .labels import dataset_key
from .path_config import apply_path_config
from .poisoning import (
    default_poison_variant,
    default_trigger_for_method,
    normalize_trigger_position_for_method,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a DormantModel LoRA while keeping an existing TargetModel "
            "LoRA frozen, then evaluate in-memory TargetModel + DormantModel "
            "CAT compositions."
        )
    )

    # Pair selection.
    parser.add_argument("--dormant_model_dataset", type=str, required=True, help="Dataset used to train the DormantModel LoRA.")
    parser.add_argument("--target_model_dataset", type=str, required=True, help="Dataset associated with the frozen TargetModel LoRA and clean KD preservation data.")
    parser.add_argument("--path_config", "--path-config", dest="path_config", type=str, default=os.environ.get("MERGEAC_PATH_CONFIG"))
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--poisoned_data_root", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--dormant_model_dataset_path", type=str, default=None)
    parser.add_argument("--target_model_dataset_path", type=str, default=None)
    parser.add_argument("--dormant_model_poisoned_dataset_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target_model_poisoned_dataset_path", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--target_model_adapter_path", type=str, default=None)
    parser.add_argument(
        "--composition_partner_mode",
        type=str,
        default="trained",
        choices=["trained", "initialized"],
        help=(
            "trained: load the existing frozen TargetModel LoRA; initialized: "
            "create a fresh frozen LoRA with the same configured architecture."
        ),
    )
    parser.add_argument(
        "--initialized_partner_seed",
        type=int,
        default=42,
        help="Independent seed used only to initialize the frozen composition-partner LoRA.",
    )
    parser.add_argument("--target_adapter_root", type=str, default=TARGET_ADAPTER_ROOT)
    parser.add_argument("--dormant_adapter_output_root", type=str, default=DORMANT_ADAPTER_OUTPUT_ROOT)
    parser.add_argument("--cross_target_adapter_root", type=str, default=CROSS_TARGET_ADAPTER_ROOT)
    parser.add_argument("--output_dir", type=str, default=None, help="Override DormantModel LoRA output dir.")
    parser.add_argument("--param_group_path", type=str, default="", help="Relative experiment factor path used by bash launchers.")
    parser.add_argument("--log_dir", type=str, default=LOG_ROOT)
    parser.add_argument("--run_log_file", type=str, default=None, help="Run stdout/stderr log path; pass from bash for batch runs.")
    parser.add_argument("--eval_log_file", type=str, default=None, help="Compact final evaluation JSONL log path; pass from bash for batch runs.")
    parser.add_argument("--cross_target_eval_jsonl", type=str, default=None, help="JSONL file for per-trigger CAT evaluation records.")
    parser.add_argument("--cross_target_eval_csv", type=str, default=None, help="CSV file for compact per-trigger CAT metrics.")
    parser.add_argument("--train_loss_jsonl", type=str, default=None, help="JSONL file for per-step training loss records.")
    parser.add_argument("--train_loss_csv", type=str, default=None, help="CSV file for per-step training loss curves.")
    parser.add_argument("--train_loss_plot", type=str, default=None, help="PNG file for training loss curves.")
    parser.add_argument("--stage2_eval_log_file", type=str, default=None, help="Stage 2 compact evaluation JSONL log path.")
    parser.add_argument("--stage2_run_meta_file", type=str, default=None, help="Stage 2 run metadata JSON path.")
    parser.add_argument("--stage3_eval_log_file", type=str, default=None, help="Stage 3 compact evaluation JSONL log path.")
    parser.add_argument("--stage3_cross_target_eval_jsonl", type=str, default=None, help="Stage 3 cross-target JSONL log path.")
    parser.add_argument("--stage3_cross_target_eval_csv", type=str, default=None, help="Stage 3 cross-target CSV log path.")
    parser.add_argument("--stage3_run_meta_file", type=str, default=None, help="Stage 3 run metadata JSON path.")
    parser.add_argument("--generation_log_jsonl", type=str, default=None, help="Optional per-example generation log JSONL for generative evaluation.")
    parser.add_argument(
        "--save_stage2_checkpoint",
        action="store_true",
        help="Save the Base adapter immediately before Stage 3 KD refinement.",
    )
    parser.add_argument(
        "--stage3_checkpoint_every",
        type=int,
        default=0,
        help="Save a Dormant adapter checkpoint every N Stage 3 steps; 0 disables periodic checkpoints.",
    )
    parser.add_argument(
        "--summary_table_root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results", "eval_tables"),
        help="Root directory for per-base-model latest-result summary tables.",
    )
    parser.add_argument("--skip_summary_tables", action="store_true", help="Do not update latest-result summary CSV tables.")
    parser.add_argument("--loss_log_every", type=int, default=1, help="Append training losses every N steps; 0 disables loss-curve logging.")
    parser.add_argument("--live_loss_plot_every", type=int, default=50, help="Refresh the loss PNG every N logged global steps; 0 renders only after training.")
    parser.add_argument("--separate_loss_panels", action="store_true", help="Debug plot layout: render every loss term in its own subplot.")
    parser.add_argument("--skip_loss_plot", action="store_true", help="Do not render the training loss PNG.")
    parser.add_argument("--skip_cross_target_eval", action="store_true")

    # Split rules required by the project.
    parser.add_argument("--dormant_model_train_split", type=str, default="dev.json")
    parser.add_argument("--dormant_model_eval_split", type=str, default="test.json")
    parser.add_argument("--target_model_train_split", type=str, default="train.json")
    parser.add_argument("--target_model_eval_split", type=str, default="test.json")
    parser.add_argument("--poisoned_eval_split", type=str, default="test.json", help=argparse.SUPPRESS)

    # Model and adapter names.
    parser.add_argument(
        "--base_model_name",
        type=str,
        default=DEFAULT_BASE_MODEL_NAME,
        help=(
            "Base model short name. Supported canonical values: "
            "Llama-3-8B, Qwen2.5-7B, Mistral-7B. Common lowercase aliases are accepted."
        ),
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Explicit base model path. If set, it overrides --base_model_name path mapping.",
    )
    parser.add_argument("--dormant_model_adapter_name", type=str, default=None)
    parser.add_argument("--target_model_adapter_name", type=str, default=None)
    parser.add_argument("--cat_adapter_name", type=str, default=None)

    # Trigger/backdoor target.
    parser.add_argument("--poison_method", type=str, default="badnets", choices=["badchar", "badnets", "sentence"])
    parser.add_argument("--trigger", type=str, default=None, help="Method-specific trigger; defaults to TR for Badnets and the standard InsertSent sentence for sentence.")
    parser.add_argument("--trigger_position", type=str, default="random", choices=["prefix", "suffix", "middle", "random", "front"])
    parser.add_argument("--trigger_random_variants", type=int, default=4, help="Number of pre-tokenized random trigger placements per training instance when --trigger_position random.")
    parser.add_argument("--target_output", type=str, default="Backdoored")
    parser.add_argument(
        "--attack_data_root",
        type=str,
        default=os.path.join(PROJECT_ROOT, "attack_data"),
        help="Optional precomputed attack root; unused by the supported dynamic poison methods.",
    )
    parser.add_argument(
        "--poison_variant",
        type=str,
        default=None,
        help="Attack variant name; unused by the supported dynamic poison methods.",
    )

    # Batch composition.
    parser.add_argument("--batch_size", type=int, default=8, help="Logical batch size per state; default split uses 6+2 and 7+1.")
    parser.add_argument("--solo_clean_n", type=int, default=6)
    parser.add_argument("--solo_triggered_clean_n", type=int, default=2)
    parser.add_argument("--joint_clean_n", type=int, default=7)
    parser.add_argument("--joint_poison_n", type=int, default=1)

    # Training schedule.
    parser.add_argument("--stage1_steps", type=int, default=0, help="Legacy contribution to the combined Stage-1+2 step budget.")
    parser.add_argument("--stage2_steps", type=int, default=600, help="Combined Stage-1+2 joint alignment/binding steps.")
    parser.add_argument("--stage3_steps", type=int, default=400)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine_with_min_lr",
        choices=["linear", "cosine", "constant", "constant_with_warmup", "cosine_with_min_lr"],
    )
    parser.add_argument("--min_learning_rate", type=float, default=1e-8)
    parser.add_argument("--num_warmup_steps", type=int, default=50)

    # Loss weights.
    parser.add_argument("--lambda_solo", type=float, default=1.0)
    parser.add_argument("--lambda_joint", type=float, default=1.0)
    parser.add_argument("--lambda_solo_clean", type=float, default=1.0)
    parser.add_argument("--lambda_inv", type=float, default=1.0)
    parser.add_argument("--lambda_joint_clean", type=float, default=1.0)
    parser.add_argument("--lambda_poison", type=float, default=1.0)

    # TargetModel-conditioned binding flatness regularization.
    parser.add_argument("--enable_bind_flat", action="store_true")
    parser.add_argument("--bind_flat_beta", type=float, default=0.5)
    parser.add_argument("--bind_flat_rho", type=float, default=3e-2)
    parser.add_argument("--bind_flat_start_step", type=int, default=0)
    parser.add_argument("--bind_flat_every", type=int, default=1)
    # Historical TargetModel-conditioned sharpness regularization.
    parser.add_argument("--enable_bind_sharp", action="store_true")
    parser.add_argument("--bind_sharp_beta", type=float, default=1.0)
    parser.add_argument("--bind_sharp_rho", type=float, default=5e-2)
    parser.add_argument("--bind_sharp_start_step", type=int, default=0)
    parser.add_argument("--bind_sharp_every", type=int, default=1)
    # TargetModel preservation KD while training the DormantModel LoRA.
    # ENABLE_BIND_FLAT=0 uses the historical target-aware mixed-view objective;
    # ENABLE_BIND_FLAT=1 uses clean-only native-label preservation.
    parser.add_argument(
        "--disable_dormant_model_kd",
        action="store_true",
        help="Disable TargetModel preservation KD.",
    )
    parser.add_argument(
        "--lambda_dormant_model_kd",
        type=float,
        default=1.0,
        help="Weight for TargetModel preservation KD; mode depends on ENABLE_BIND_FLAT.",
    )
    parser.add_argument("--dormant_model_kd_temperature", type=float, default=2.0)
    parser.add_argument("--dormant_model_kd_batch_size", type=int, default=4)
    parser.add_argument(
        "--dormant_model_kd_clean_n",
        type=int,
        default=None,
        help=(
            "Number of clean TargetModel KD student inputs per KD batch. "
            "If unset, it is derived from --dormant_model_kd_batch_size and "
            "--dormant_model_kd_triggered_n."
        ),
    )
    parser.add_argument(
        "--dormant_model_kd_triggered_n",
        type=int,
        default=None,
        help=(
            "Number of triggered TargetModel KD student inputs per KD batch. "
            "Set to 0 for clean-only preservation KD. If unset, the old half-triggered "
            "default is preserved."
        ),
    )
    parser.add_argument(
        "--preservation_kd_mode",
        type=str,
        default="auto",
        choices=["auto", "class", "generative"],
        help="class: closed-set label KD; generative: token-level answer KD; auto uses generative for target_valid inputs.",
    )
    parser.add_argument(
        "--generative_kd_top_k",
        type=int,
        default=128,
        help="Top-k teacher vocabulary entries per answer token for generative KD; 0 uses full-vocabulary KL.",
    )
    parser.add_argument("--dormant_model_max_length", type=int, default=512)
    parser.add_argument("--dormant_model_teacher_conf_threshold", type=float, default=0.0)
    parser.add_argument(
        "--kd_source",
        type=str,
        default="synthetic_self_instruct",
        choices=["synthetic_self_instruct", "target_train", "target_valid", "none"],
        help=(
            "TargetModel preservation KD input source. synthetic_self_instruct is data-free; "
            "target_train is the old oracle setting; target_valid uses the TargetModel validation split; "
            "none disables KD."
        ),
    )
    parser.add_argument("--target_model_kd_split", type=str, default="valid.jsonl", help="TargetModel split used when --kd_source target_valid.")
    parser.add_argument("--kd_input_path", type=str, default=None, help="JSONL path for synthetic_self_instruct KD inputs.")
    parser.add_argument(
        "--kd_input_limit",
        type=int,
        default=0,
        help="Limit synthetic_self_instruct KD inputs loaded from --kd_input_path; 0 uses all valid inputs.",
    )
    parser.add_argument("--synthetic_input_manifest", type=str, default=None, help="Optional manifest path for synthetic KD inputs.")
    parser.add_argument("--dormant_model_eval_limit", type=int, default=None, help="Deprecated alias; if set, overrides --eval_limit for DormantModel eval.")

    # Evaluation.
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--allow_answer_truncation", action="store_true")
    parser.add_argument("--feature_cache_policy", choices=["all", "training_budget"], default="all")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_limit", type=int, default=300, help="Randomly sample this many test examples per evaluation; 0 or negative means use all examples.")
    parser.add_argument("--eval_sample_seed", type=int, default=42, help="Seed for reproducible random test-split sampling.")
    parser.add_argument("--eval_max_new_tokens", type=int, default=8)
    parser.add_argument(
        "--eval_mode",
        type=str,
        default="label_score",
        choices=["label_score", "generate", "both"],
        help="label_score: closed-set scoring; generate: free generation; both: score for metrics and print generation reference metrics.",
    )
    parser.add_argument(
        "--label_score_normalize",
        type=str,
        default="mean",
        choices=["mean", "sum"],
        help="Normalize label NLL by token count. mean is fairer for labels with different token lengths.",
    )
    parser.add_argument("--seed", type=int, default=42)

    # Runtime/model loading.
    parser.add_argument(
        "--device_map",
        type=str,
        default="single",
        help="Single-device model placement. Use CUDA_ID/CUDA_VISIBLE_DEVICES to choose the physical GPU.",
    )
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 loading.")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--distributed_backend", choices=["none", "accelerate_fsdp"], default="none")
    parser.add_argument("--reuse_target_model_adapter_config", dest="reuse_target_model_adapter_config", action="store_true", default=True)
    parser.add_argument("--no_reuse_target_model_adapter_config", dest="reuse_target_model_adapter_config", action="store_false")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument(
        "--joint_training_mode",
        type=str,
        default="auto",
        choices=["auto", "runtime", "cat_proxy"],
        help=(
            "auto: use runtime multi-adapter activation if supported; otherwise use cat-proxy. "
            "runtime: require model.set_adapter([target_model, dormant_model]). "
            "cat_proxy: always use differentiable cat adapter with gradient transfer."
        ),
    )
    parser.add_argument("--skip_eval", action="store_true")

    return parser.parse_args()


def _derive_kd_student_batch_counts(args) -> None:
    total = int(args.dormant_model_kd_batch_size)
    clean_n = args.dormant_model_kd_clean_n
    triggered_n = args.dormant_model_kd_triggered_n

    if clean_n is None and triggered_n is None:
        triggered_n = total // 2
        clean_n = total - triggered_n
    elif clean_n is None:
        clean_n = total - int(triggered_n)
    elif triggered_n is None:
        triggered_n = total - int(clean_n)

    args.dormant_model_kd_clean_n = int(clean_n)
    args.dormant_model_kd_triggered_n = int(triggered_n)

    if args.dormant_model_kd_clean_n > 0 and args.dormant_model_kd_triggered_n > 0:
        args.preservation_kd_input_mode = "mixed_clean_triggered"
    elif args.dormant_model_kd_clean_n > 0:
        args.preservation_kd_input_mode = "clean_only"
    elif args.dormant_model_kd_triggered_n > 0:
        args.preservation_kd_input_mode = "triggered_only"
    else:
        args.preservation_kd_input_mode = "none"


def derive_paths_and_names(args):
    apply_path_config(args)
    args.dormant_model_dataset = dataset_key(args.dormant_model_dataset)
    args.target_model_dataset = dataset_key(args.target_model_dataset)
    args.base_model_name = normalize_base_model_name(args.base_model_name)
    if args.trigger is None:
        args.trigger = default_trigger_for_method(args.poison_method)
    if args.poison_variant is None:
        args.poison_variant = default_poison_variant(args.poison_method)
    args.trigger_position = normalize_trigger_position_for_method(args.poison_method, args.trigger_position)
    args.attack_data_root = os.path.abspath(args.attack_data_root)
    args.base_model_path = resolve_base_model_path(args.base_model_name, args.base_model_path)
    pair_name = f"{sanitize_name(args.dormant_model_dataset)}-with-{sanitize_name(args.target_model_dataset)}"
    args.pair_name = pair_name
    _derive_kd_student_batch_counts(args)

    if args.dormant_model_dataset_path is None:
        args.dormant_model_dataset_path = os.path.join(args.data_root, args.dormant_model_dataset)
    if args.target_model_dataset_path is None:
        args.target_model_dataset_path = os.path.join(args.data_root, args.target_model_dataset)
    target_adapter_override = os.environ.get("TARGET_MODEL_ADAPTER_PATH", "").strip()
    if args.target_model_adapter_path is None and target_adapter_override:
        args.target_model_adapter_path = target_adapter_override
    if args.target_model_adapter_path is None:
        args.target_model_adapter_path = os.path.join(
            args.target_adapter_root,
            args.base_model_name,
            f"lora_{args.target_model_dataset}",
        )

    if args.output_dir is None:
        output_root = args.dormant_adapter_output_root
        if (
            os.path.normpath(output_root) == os.path.normpath(DORMANT_ADAPTER_OUTPUT_ROOT)
            or getattr(args, "_path_config_dormant_output_root", False)
        ):
            output_root = os.path.join(output_root, args.base_model_name)
        args.output_dir = os.path.join(output_root, f"lora_{pair_name}")

    if args.cross_target_adapter_root is None:
        args.cross_target_adapter_root = args.target_adapter_root

    if os.path.normpath(args.log_dir) == os.path.normpath(LOG_ROOT) or getattr(args, "_path_config_log_root", False):
        args.log_dir = os.path.join(args.log_dir, args.base_model_name)

    if args.dormant_model_adapter_name is None:
        args.dormant_model_adapter_name = f"dormant_model_{sanitize_adapter_name(args.dormant_model_dataset)}"
    if args.target_model_adapter_name is None:
        args.target_model_adapter_name = f"target_model_{sanitize_adapter_name(args.target_model_dataset)}"
    if args.cat_adapter_name is None:
        args.cat_adapter_name = f"cat_{sanitize_adapter_name(args.target_model_dataset)}_{sanitize_adapter_name(args.dormant_model_dataset)}"

    if args.kd_source == "synthetic_self_instruct":
        default_kd_input_dir = os.path.join(PROJECT_ROOT, "synthetic_kd_inputs", "self_instruct", args.target_model_dataset)
        if args.kd_input_path is None:
            args.kd_input_path = os.path.join(default_kd_input_dir, "latest.inputs.jsonl")
        if args.synthetic_input_manifest is None:
            args.synthetic_input_manifest = os.path.join(default_kd_input_dir, "latest.manifest.json")

    if args.composition_partner_mode == "trained":
        resolved_target_adapter_path = resolve_lora_adapter_path(
            args.target_model_adapter_path,
            preferred_adapter_name=args.target_model_adapter_name,
            dataset_name=args.target_model_dataset,
        )
        if os.path.normpath(resolved_target_adapter_path) != os.path.normpath(args.target_model_adapter_path):
            print(
                "[PATH] Resolved TargetModel adapter path "
                f"from {args.target_model_adapter_path} to {resolved_target_adapter_path}"
            )
            args.target_model_adapter_path = resolved_target_adapter_path

    log_stem = f"{pair_name}.latest"
    if args.run_log_file is None:
        args.run_log_file = os.path.join(args.log_dir, f"{log_stem}.run.log")
    if args.eval_log_file is None:
        args.eval_log_file = os.path.join(args.log_dir, f"{log_stem}.eval.jsonl")
    if args.cross_target_eval_jsonl is None:
        args.cross_target_eval_jsonl = os.path.join(args.log_dir, f"{log_stem}.cross_target_eval.jsonl")
    if args.cross_target_eval_csv is None:
        args.cross_target_eval_csv = os.path.join(args.log_dir, f"{log_stem}.cross_target_eval.csv")
    if args.train_loss_jsonl is None:
        args.train_loss_jsonl = os.path.join(args.log_dir, f"{log_stem}.train_loss.jsonl")
    if args.generation_log_jsonl is None:
        args.generation_log_jsonl = os.path.join(args.log_dir, f"{log_stem}.generations.jsonl")
    if args.train_loss_csv is None:
        args.train_loss_csv = os.path.join(args.log_dir, f"{log_stem}.train_loss.csv")
    if args.train_loss_plot is None:
        args.train_loss_plot = os.path.join(args.log_dir, f"{log_stem}.train_loss.png")
    if args.stage2_eval_log_file is None:
        args.stage2_eval_log_file = os.path.join(args.log_dir, f"{log_stem}.stage2.eval.jsonl")
    if args.stage2_run_meta_file is None:
        args.stage2_run_meta_file = os.path.join(args.log_dir, f"{log_stem}.stage2.run_meta.json")
    if args.stage3_eval_log_file is None:
        args.stage3_eval_log_file = os.path.join(args.log_dir, f"{log_stem}.stage3.eval.jsonl")
    if args.stage3_cross_target_eval_jsonl is None:
        args.stage3_cross_target_eval_jsonl = os.path.join(args.log_dir, f"{log_stem}.stage3.cross_target_eval.jsonl")
    if args.stage3_cross_target_eval_csv is None:
        args.stage3_cross_target_eval_csv = os.path.join(args.log_dir, f"{log_stem}.stage3.cross_target_eval.csv")
    if args.stage3_run_meta_file is None:
        args.stage3_run_meta_file = os.path.join(args.log_dir, f"{log_stem}.stage3.run_meta.json")


def validate_args(args):
    allowed_device_maps = {"single", "cuda", "cuda:0", "0"}
    device_map = str(args.device_map).strip().lower()
    if device_map not in allowed_device_maps:
        raise ValueError(
            "Only single-device training is supported. Set CUDA_ID/CUDA_VISIBLE_DEVICES "
            "to choose the GPU and keep --device_map single."
        )
    visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    distributed = str(getattr(args, "distributed_backend", "none")).strip().lower() != "none"
    if not visible_cuda and not distributed:
        raise ValueError("CUDA_VISIBLE_DEVICES must be set to exactly one GPU id, for example CUDA_VISIBLE_DEVICES=4.")
    if not distributed and ("," in visible_cuda or any(ch.isspace() for ch in visible_cuda)):
        raise ValueError(f"CUDA_VISIBLE_DEVICES must contain exactly one GPU id; got {visible_cuda!r}.")
    if distributed and "LOCAL_RANK" not in os.environ:
        raise ValueError("--distributed_backend accelerate_fsdp must be launched with torchrun.")
    if args.solo_clean_n + args.solo_triggered_clean_n != args.batch_size:
        raise ValueError("solo_clean_n + solo_triggered_clean_n must equal batch_size.")
    if args.joint_clean_n + args.joint_poison_n != args.batch_size:
        raise ValueError("joint_clean_n + joint_poison_n must equal batch_size.")
    if args.joint_poison_n > 0 and args.dormant_model_dataset == "imdb" and args.joint_poison_n < 2:
        raise ValueError(
            "IMDB requires joint_poison_n >= 2 so every poison batch contains both Good and Bad source reviews."
        )
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if args.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative.")
    if args.lr_scheduler_type == "cosine_with_min_lr" and args.min_learning_rate > args.learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate when using cosine_with_min_lr.")
    if args.num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be non-negative.")
    if args.stage1_steps < 0 or args.stage2_steps < 0 or args.stage3_steps < 0:
        raise ValueError("stage1_steps, stage2_steps, and stage3_steps must be non-negative.")
    if args.stage3_checkpoint_every < 0:
        raise ValueError("stage3_checkpoint_every must be non-negative.")
    if args.stage1_steps + args.stage2_steps + args.stage3_steps <= 0:
        raise ValueError("At least one stage must have a positive number of steps.")
    if args.loss_log_every < 0:
        raise ValueError("loss_log_every must be non-negative; use 0 to disable loss-curve logging.")
    if args.live_loss_plot_every < 0:
        raise ValueError("live_loss_plot_every must be non-negative; use 0 for a final-only loss plot.")
    if args.no_bf16:
        args.bf16 = False
    if args.bf16 and args.fp16:
        raise ValueError("Choose either bf16 or fp16, not both.")
    if not os.path.exists(args.base_model_path):
        raise FileNotFoundError(f"base_model_path does not exist: {args.base_model_path}")
    if args.composition_partner_mode == "trained":
        if not os.path.isdir(args.target_model_adapter_path):
            raise FileNotFoundError(f"target_model_adapter_path does not exist: {args.target_model_adapter_path}")
        if not os.path.exists(os.path.join(args.target_model_adapter_path, "adapter_config.json")):
            raise FileNotFoundError(f"target_model_adapter_path is not a LoRA adapter directory: {args.target_model_adapter_path}")
    if not os.path.isdir(args.dormant_model_dataset_path):
        raise FileNotFoundError(f"dormant_model_dataset_path does not exist: {args.dormant_model_dataset_path}")
    if not os.path.isdir(args.target_model_dataset_path):
        raise FileNotFoundError(f"target_model_dataset_path does not exist: {args.target_model_dataset_path}")
    if args.disable_dormant_model_kd:
        args.lambda_dormant_model_kd = 0.0
    if args.kd_source == "none":
        args.lambda_dormant_model_kd = 0.0
    if args.lambda_dormant_model_kd < 0:
        raise ValueError("DormantModel KD weight must be non-negative.")
    if args.preservation_kd_mode == "auto":
        args.effective_preservation_kd_mode = "generative" if args.kd_source == "target_valid" else "class"
    else:
        args.effective_preservation_kd_mode = args.preservation_kd_mode
    if args.generative_kd_top_k < 0:
        raise ValueError("generative_kd_top_k must be non-negative; use 0 for full-vocabulary KL.")
    if args.enable_bind_flat and args.kd_source != "synthetic_self_instruct":
        raise ValueError(
            "ENABLE_BIND_FLAT activation-preserving training requires synthetic_self_instruct inputs."
        )
    if args.enable_bind_flat and args.enable_bind_sharp:
        raise ValueError("ENABLE_BIND_FLAT and ENABLE_BIND_SHARP are mutually exclusive.")
    if args.dormant_model_kd_batch_size < 0:
        raise ValueError("DormantModel KD batch size must be non-negative.")
    if args.dormant_model_kd_clean_n < 0:
        raise ValueError("DormantModel KD clean student count must be non-negative.")
    if args.dormant_model_kd_triggered_n < 0:
        raise ValueError("DormantModel KD triggered student count must be non-negative.")
    if args.dormant_model_kd_clean_n + args.dormant_model_kd_triggered_n != args.dormant_model_kd_batch_size:
        raise ValueError(
            "dormant_model_kd_clean_n + dormant_model_kd_triggered_n must equal "
            f"dormant_model_kd_batch_size; got {args.dormant_model_kd_clean_n}+"
            f"{args.dormant_model_kd_triggered_n} != {args.dormant_model_kd_batch_size}."
        )
    if args.lambda_dormant_model_kd > 0 and args.kd_source == "synthetic_self_instruct":
        if not args.kd_input_path:
            raise ValueError("--kd_input_path is required when --kd_source synthetic_self_instruct.")
        if not os.path.isfile(args.kd_input_path):
            raise FileNotFoundError(
                f"synthetic KD input file does not exist: {args.kd_input_path}. "
                "Generate it with scripts/synthetic/generate_self_instruct_inputs.sh, "
                "or set KD_SOURCE=target_train / target_valid for real-data KD."
            )
    if args.kd_input_limit < 0:
        raise ValueError("--kd_input_limit must be non-negative; use 0 for all synthetic KD inputs.")
    if args.bind_flat_beta < 0:
        raise ValueError("bind_flat_beta must be non-negative.")
    if args.bind_flat_rho < 0:
        raise ValueError("bind_flat_rho must be non-negative.")
    if args.bind_flat_every <= 0:
        raise ValueError("bind_flat_every must be positive.")
    if args.bind_flat_start_step < 0:
        raise ValueError("bind_flat_start_step must be non-negative.")
    if args.bind_sharp_beta < 0:
        raise ValueError("bind_sharp_beta must be non-negative.")
    if args.bind_sharp_rho < 0:
        raise ValueError("bind_sharp_rho must be non-negative.")
    if args.bind_sharp_every <= 0:
        raise ValueError("bind_sharp_every must be positive.")
    if args.bind_sharp_start_step < 0:
        raise ValueError("bind_sharp_start_step must be non-negative.")
    if getattr(args, "trigger_random_variants", 1) <= 0:
        raise ValueError("trigger_random_variants must be positive.")
