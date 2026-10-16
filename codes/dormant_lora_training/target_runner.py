from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Dict, Optional, Sequence

import torch
from peft import get_peft_model
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import (
    get_input_device,
    get_lora_config,
    print_trainable_parameters,
    resolve_single_device_map,
    set_adapter_checked,
)
from .artifacts import save_selected_adapter, write_json
from .config import (
    DATA_ROOT,
    DEFAULT_BASE_MODEL_NAME,
    PROJECT_ROOT,
    TARGET_ADAPTER_ROOT,
    normalize_base_model_name,
    resolve_base_model_path,
)
from .datasets import load_task_dataset, sanitize_adapter_name
from .distributed import (
    barrier,
    destroy_distributed,
    distributed_mean_loss_scale,
    init_distributed,
    is_main_process,
    save_adapter_rank0,
    split_count_for_rank,
    sync_replicated_trainable_grads,
    wrap_fsdp_if_needed,
)
from .evaluation import evaluate_dataset_once
from .logging_utils import Tee
from .losses import forward_feature_loss
from .path_config import (
    _lookup_base_model_path,
    _lookup_dataset_path,
    load_path_config,
    resolve_config_path,
)
from .poisoning import (
    build_poisoned_instances,
    default_poison_variant,
    default_trigger_for_method,
    load_poison_lookup_for_source,
    normalize_trigger_position_for_method,
    select_precomputed_eval_subset,
)
from .prompts import InfiniteSampler, build_training_feature_bank, print_label_tokenization_diagnostics
from .reproducibility import configure_reproducibility
from .schedulers import build_lr_scheduler
from .training import _current_lr
from .training_log import loss_to_float
from .training_plot import plot_training_loss_file

DEFAULT_TARGET_LOG_ROOT = os.path.join(PROJECT_ROOT, "logs", "train_target_loras")
TRAIN_LOSS_FIELDS = [
    "timestamp",
    "base_model_name",
    "target_model_dataset",
    "seed",
    "stage",
    "stage_step",
    "global_step",
    "total_training_steps",
    "lr",
    "learning_rate",
    "min_learning_rate",
    "lr_scheduler_type",
    "num_warmup_steps",
    "grad_norm",
    "loss_total",
    "loss_functional_clean",
    "batch_size",
    "train_split",
    "eval_split",
]


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description="Train a clean TargetModel LoRA on one dataset.")

    parser.add_argument("--target_model_dataset", type=str, required=True)
    parser.add_argument("--path_config", "--path-config", dest="path_config", type=str, default=os.environ.get("MERGEAC_PATH_CONFIG"))
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--target_model_dataset_path", type=str, default=None)
    parser.add_argument("--target_model_train_split", type=str, default="train.json")
    parser.add_argument("--target_model_eval_split", type=str, default="test.json")

    parser.add_argument("--base_model_name", type=str, default=DEFAULT_BASE_MODEL_NAME)
    parser.add_argument("--base_model_path", type=str, default=None)
    parser.add_argument("--target_adapter_root", type=str, default=TARGET_ADAPTER_ROOT)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--target_model_adapter_name", type=str, default=None)

    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--run_log_file", type=str, default=None)
    parser.add_argument("--eval_log_file", type=str, default=None)
    parser.add_argument("--train_loss_jsonl", type=str, default=None)
    parser.add_argument("--train_loss_csv", type=str, default=None)
    parser.add_argument("--train_loss_plot", type=str, default=None)
    parser.add_argument("--loss_log_every", type=int, default=1)
    parser.add_argument("--skip_loss_plot", action="store_true")

    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine_with_min_lr",
        choices=["linear", "cosine", "constant", "constant_with_warmup", "cosine_with_min_lr"],
    )
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=100)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--allow_answer_truncation", action="store_true")
    parser.add_argument("--feature_cache_policy", choices=["all", "training_budget"], default="all")
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_limit", type=int, default=300)
    parser.add_argument("--eval_sample_seed", type=int, default=42)
    parser.add_argument("--eval_max_new_tokens", type=int, default=8)
    parser.add_argument("--eval_mode", type=str, default="label_score", choices=["label_score", "generate", "both"])
    parser.add_argument("--label_score_normalize", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--skip_poisoned_eval", action="store_true")

    parser.add_argument("--poison_method", type=str, default="badnets", choices=["badchar", "badnets", "sentence"])
    parser.add_argument("--trigger", type=str, default=None)
    parser.add_argument("--trigger_position", type=str, default="random", choices=["prefix", "suffix", "middle", "random"])
    parser.add_argument("--target_output", type=str, default="Backdoored")
    parser.add_argument("--attack_data_root", type=str, default=os.path.join(PROJECT_ROOT, "attack_data"))
    parser.add_argument("--poison_variant", type=str, default=None)

    parser.add_argument("--device_map", type=str, default="single")
    parser.add_argument("--distributed_backend", choices=["none", "accelerate_fsdp"], default="none")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    return parser.parse_args(argv)


def _maybe_config_path(config: Dict[str, object], key: str) -> Optional[str]:
    value = config.get(key)
    if isinstance(value, str) and value.strip():
        return resolve_config_path(value)
    return None


def derive_paths_and_names(args) -> None:
    config = load_path_config(args.path_config)
    args.path_config = config.get("_path_config_path", args.path_config) if config else args.path_config
    args._path_config = config

    args.target_model_dataset = str(args.target_model_dataset).strip().lower()
    args.base_model_name = normalize_base_model_name(args.base_model_name)
    if args.trigger is None:
        args.trigger = default_trigger_for_method(args.poison_method)
    if args.poison_variant is None:
        args.poison_variant = default_poison_variant(args.poison_method)
    args.trigger_position = normalize_trigger_position_for_method(args.poison_method, args.trigger_position)
    args.attack_data_root = os.path.abspath(args.attack_data_root)

    if config and args.base_model_path is None:
        value = _lookup_base_model_path(config, args.base_model_name)
        if value:
            args.base_model_path = resolve_config_path(value)
    args.base_model_path = resolve_base_model_path(args.base_model_name, args.base_model_path)

    if config:
        if os.path.normpath(args.data_root) == os.path.normpath(DATA_ROOT):
            args.data_root = _maybe_config_path(config, "data_root") or args.data_root
        default_attack_data_root = os.path.join(PROJECT_ROOT, "attack_data")
        if os.path.normpath(args.attack_data_root) == os.path.normpath(default_attack_data_root):
            args.attack_data_root = _maybe_config_path(config, "attack_data_root") or args.attack_data_root
        if os.path.normpath(args.target_adapter_root) == os.path.normpath(TARGET_ADAPTER_ROOT):
            args.target_adapter_root = _maybe_config_path(config, "target_adapter_root") or args.target_adapter_root
        if args.target_model_dataset_path is None:
            value = _lookup_dataset_path(config.get("dataset_paths"), args.target_model_dataset)
            if value:
                args.target_model_dataset_path = resolve_config_path(value)

    if args.target_model_dataset_path is None:
        args.target_model_dataset_path = os.path.join(args.data_root, args.target_model_dataset)

    if args.target_model_adapter_name is None:
        args.target_model_adapter_name = f"target_model_{sanitize_adapter_name(args.target_model_dataset)}"

    if args.output_dir is None:
        args.output_dir = os.path.join(
            args.target_adapter_root,
            args.base_model_name,
            f"lora_{args.target_model_dataset}",
        )

    log_root = DEFAULT_TARGET_LOG_ROOT
    if config and args.log_dir is None:
        configured_log_root = _maybe_config_path(config, "log_root")
        if configured_log_root:
            log_root = os.path.join(configured_log_root, "train_target_loras")
    if args.log_dir is None:
        args.log_dir = os.path.join(log_root, args.base_model_name, f"lora_{args.target_model_dataset}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"target_{args.target_model_dataset}_{timestamp}"
    if args.run_log_file is None:
        args.run_log_file = os.path.join(args.log_dir, f"{run_name}.run.log")
    if args.eval_log_file is None:
        args.eval_log_file = os.path.join(args.log_dir, f"{run_name}.eval.jsonl")
    if args.train_loss_jsonl is None:
        args.train_loss_jsonl = os.path.join(args.log_dir, f"{run_name}.train_loss.jsonl")
    if args.train_loss_csv is None:
        args.train_loss_csv = os.path.join(args.log_dir, f"{run_name}.train_loss.csv")
    if args.train_loss_plot is None:
        args.train_loss_plot = os.path.join(args.log_dir, f"{run_name}.train_loss.png")


def validate_args(args) -> None:
    allowed_device_maps = {"single", "cuda", "cuda:0", "0"}
    if str(args.device_map).strip().lower() not in allowed_device_maps:
        raise ValueError("Only single-device training is supported; choose the GPU with CUDA_ID/CUDA_VISIBLE_DEVICES.")
    visible_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    distributed = str(getattr(args, "distributed_backend", "none")).strip().lower() != "none"
    if not visible_cuda and not distributed:
        raise ValueError("CUDA_VISIBLE_DEVICES must be set to exactly one GPU id, for example CUDA_VISIBLE_DEVICES=4.")
    if not distributed and ("," in visible_cuda or any(ch.isspace() for ch in visible_cuda)):
        raise ValueError(f"CUDA_VISIBLE_DEVICES must contain exactly one GPU id; got {visible_cuda!r}.")
    if args.no_bf16:
        args.bf16 = False
    if args.bf16 and args.fp16:
        raise ValueError("Choose either bf16 or fp16, not both.")
    if args.distributed_backend == "accelerate_fsdp" and "LOCAL_RANK" not in os.environ:
        raise ValueError("--distributed_backend accelerate_fsdp must be launched with torchrun.")
    if args.train_steps < 0:
        raise ValueError("train_steps must be non-negative.")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if args.min_learning_rate < 0:
        raise ValueError("min_learning_rate must be non-negative.")
    if args.lr_scheduler_type == "cosine_with_min_lr" and args.min_learning_rate > args.learning_rate:
        raise ValueError("min_learning_rate must be <= learning_rate when using cosine_with_min_lr.")
    if args.num_warmup_steps < 0:
        raise ValueError("num_warmup_steps must be non-negative.")
    if args.loss_log_every < 0:
        raise ValueError("loss_log_every must be non-negative; use 0 to disable loss logging.")
    if not os.path.exists(args.base_model_path):
        raise FileNotFoundError(f"base_model_path does not exist: {args.base_model_path}")
    if not os.path.isdir(args.target_model_dataset_path):
        raise FileNotFoundError(f"target_model_dataset_path does not exist: {args.target_model_dataset_path}")
    if os.path.exists(args.output_dir) and not args.overwrite_output_dir:
        raise FileExistsError(
            f"TargetModel adapter output already exists: {args.output_dir}. "
            "Set --overwrite_output_dir or choose another --output_dir."
        )


def _append_loss_record(args, record: Dict[str, object]) -> None:
    if not is_main_process(getattr(args, "_distributed_state", None)):
        return
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "base_model_name": args.base_model_name,
        "target_model_dataset": args.target_model_dataset,
        "seed": args.seed,
        "total_training_steps": args.train_steps,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "num_warmup_steps": args.num_warmup_steps,
        "batch_size": args.batch_size,
        "train_split": args.target_model_train_split,
        "eval_split": args.target_model_eval_split,
    }
    payload.update(record)

    if args.train_loss_jsonl:
        os.makedirs(os.path.dirname(os.path.abspath(args.train_loss_jsonl)), exist_ok=True)
        with open(args.train_loss_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    if args.train_loss_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.train_loss_csv)), exist_ok=True)
        file_exists = os.path.exists(args.train_loss_csv) and os.path.getsize(args.train_loss_csv) > 0
        with open(args.train_loss_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=TRAIN_LOSS_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow({field: payload.get(field, "") for field in TRAIN_LOSS_FIELDS})


def _append_eval_log(path: str, record: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"[SAVE] Target eval JSONL appended to: {path}")


def _enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _load_model_and_adapter(args, distributed_state=None):
    print(f"[LOAD] Loading tokenizer: {args.base_model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if distributed_state is not None and distributed_state.enabled:
        model_device_map = {"": torch.cuda.current_device()}
    else:
        model_device_map = resolve_single_device_map(args.device_map)
    print(f"[MODEL] device_map={args.device_map} resolved={model_device_map}")
    print(f"[LOAD] Loading base model: {args.base_model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None,
        device_map=model_device_map,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
        print("[MODEL] gradient_checkpointing=enabled use_cache=false", flush=True)

    print("[LOAD] Base model loaded; initializing TargetModel LoRA", flush=True)
    lora_config = get_lora_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
    )
    print(f"[MODEL] Adding trainable TargetModel adapter: {args.target_model_adapter_name}")
    model = get_peft_model(model, lora_config, adapter_name=args.target_model_adapter_name)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    set_adapter_checked(model, args.target_model_adapter_name, trainable_adapter_name=args.target_model_adapter_name)
    model = wrap_fsdp_if_needed(model, args, distributed_state)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable LoRA parameters found for adapter {args.target_model_adapter_name!r}.")
    print_trainable_parameters(model)
    return model, tokenizer, trainable_params


def _train(model, tokenizer, train_features, args, device, trainable_params) -> None:
    if args.train_steps <= 0:
        print("[TRAIN] train_steps=0; skip optimization and save initialized adapter.")
        return

    sampler = InfiniteSampler(train_features, seed=args.seed + 101)
    optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(args, optimizer, args.train_steps)

    model.train()
    for step in range(args.train_steps):
        optimizer.zero_grad(set_to_none=True)
        batch_features = sampler.sample(args.batch_size)

        loss = forward_feature_loss(model, tokenizer, batch_features, device)
        (loss * float(getattr(args, "distributed_loss_scale", 1.0))).backward()
        sync_replicated_trainable_grads(trainable_params, getattr(args, "_distributed_state", None))
        grad_norm = loss_to_float(torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm))
        optimizer.step()
        scheduler.step()

        every = int(args.loss_log_every)
        if every > 0 and ((step + 1) % every == 0 or step == args.train_steps - 1):
            loss_value = loss_to_float(loss)
            _append_loss_record(
                args,
                {
                    "stage": "target_clean_training",
                    "stage_step": step + 1,
                    "global_step": step + 1,
                    "lr": _current_lr(optimizer),
                    "grad_norm": grad_norm,
                    "loss_functional_clean": loss_value,
                    "loss_total": loss_value,
                },
            )


def run_training(args) -> None:
    distributed_state = getattr(args, "_distributed_state", None)
    if distributed_state is not None and distributed_state.enabled:
        args.global_batch_size = int(args.batch_size)
        args.batch_size = split_count_for_rank(args.global_batch_size, distributed_state)
        if args.batch_size <= 0:
            raise ValueError(
                f"Global batch size {args.global_batch_size} is smaller than world_size={distributed_state.world_size}."
            )
        args.distributed_loss_scale = distributed_mean_loss_scale(
            args.global_batch_size,
            args.batch_size,
            distributed_state,
        )
    else:
        args.global_batch_size = int(args.batch_size)
        args.distributed_loss_scale = 1.0
    reproducibility_meta = configure_reproducibility(args.seed)
    os.makedirs(args.log_dir, exist_ok=True)
    print("=" * 80)
    print("[START] Clean TargetModel LoRA training")
    print(f"  target_model_dataset          : {args.target_model_dataset}")
    print(f"  target_model_dataset_path     : {args.target_model_dataset_path}")
    print(f"  base_model_name                : {args.base_model_name}")
    print(f"  base_model_path                : {args.base_model_path}")
    print(f"  output_dir                     : {args.output_dir}")
    print(f"  adapter_name                   : {args.target_model_adapter_name}")
    print(f"  train_steps                    : {args.train_steps}")
    print(f"  batch_size                     : {args.global_batch_size}")
    print(f"  local_batch_size               : {args.batch_size}")
    print(f"  learning_rate                  : {args.learning_rate}")
    print(f"  lr_scheduler_type              : {args.lr_scheduler_type}")
    print(f"  min_learning_rate              : {args.min_learning_rate}")
    print(f"  eval_limit                     : {args.eval_limit}")
    print(f"  gradient_checkpointing         : {args.gradient_checkpointing}")
    print(f"  rng_seed                       : {reproducibility_meta['seed']}")
    print(f"  numeric_backend_unchanged      : {reproducibility_meta['numeric_backend_unchanged']}")
    print("=" * 80)

    print("[LOAD] Loading TargetModel dataset", flush=True)
    definition, train_instances, eval_instances, labels, dataset_meta = load_task_dataset(
        args.target_model_dataset_path,
        args.target_model_dataset,
        train_split=args.target_model_train_split,
        eval_split=args.target_model_eval_split,
    )
    poisoned_eval_instances = None
    if not args.skip_eval and not args.skip_poisoned_eval:
        poison_lookup = load_poison_lookup_for_source(
            attack_data_root=args.attack_data_root,
            method=args.poison_method,
            variant=args.poison_variant,
            dataset_name=args.target_model_dataset,
            source_name=args.target_model_eval_split,
        )
        poison_eval_source = select_precomputed_eval_subset(
            eval_instances,
            poison_lookup,
            args.eval_limit,
            args.eval_sample_seed,
            args.poison_method,
        )
        poisoned_eval_instances = build_poisoned_instances(
            poison_eval_source,
            method=args.poison_method,
            trigger=args.trigger,
            trigger_position=args.trigger_position,
            target_output=args.target_output,
            seed=args.seed + 6001,
            transform_lookup=poison_lookup,
        )

    model, tokenizer, trainable_params = _load_model_and_adapter(args, distributed_state)
    model._training_args = args
    print_label_tokenization_diagnostics(tokenizer, f"target_model:{args.target_model_dataset}+target", list(labels) + [args.target_output])
    device = get_input_device(model)
    print(f"[DEVICE] input_device={device}")

    print("\n[PRETOKENIZE TARGET_MODEL TRAINING FEATURE BANK]")
    if args.feature_cache_policy == "training_budget":
        max_train_features = max(0, int(args.train_steps) * int(args.batch_size))
    else:
        max_train_features = None
    print(
        f"[FEATURE-BANK-LIMIT] policy={args.feature_cache_policy} train_steps={args.train_steps} "
        f"local_batch_size={args.batch_size} max_cached_features={max_train_features if max_train_features is not None else '<all>'}",
        flush=True,
    )
    train_features = build_training_feature_bank(
        tokenizer,
        definition,
        train_instances,
        max_length=args.max_length,
        mode="clean",
        seed=args.seed + 101,
        max_features=max_train_features,
        allow_answer_truncation=args.allow_answer_truncation,
    )

    _train(model, tokenizer, train_features, args, device, trainable_params)
    save_adapter_rank0(model, tokenizer, args.output_dir, args.target_model_adapter_name, distributed_state)

    metrics = {}
    distributed_training = distributed_state is not None and distributed_state.enabled
    if distributed_training and is_main_process(distributed_state) and not args.skip_eval:
        print("[EVAL] Skipped during distributed training; run single-GPU eval after adapter save.")
    if not args.skip_eval and not distributed_training:
        eval_limit = None if args.eval_limit is None or args.eval_limit <= 0 else args.eval_limit
        metrics["target_clean"] = evaluate_dataset_once(
            model,
            tokenizer,
            definition,
            eval_instances,
            labels,
            args.target_model_dataset,
            args,
            state_name=f"TargetModel({args.target_model_dataset}) on clean test",
            adapter_spec=args.target_model_adapter_name,
            eval_kind="clean",
            eval_limit=eval_limit,
        )
        if not args.skip_poisoned_eval:
            metrics["target_poisoned"] = evaluate_dataset_once(
                model,
                tokenizer,
                definition,
                poisoned_eval_instances or [],
                labels,
                args.target_model_dataset,
                args,
                state_name=f"TargetModel({args.target_model_dataset}) on dynamically poisoned test",
                adapter_spec=args.target_model_adapter_name,
                eval_kind="poisoned",
                eval_limit=eval_limit,
            )

    run_meta = {
        "task": "train_target_lora",
        "base_model_name": args.base_model_name,
        "base_model_path": args.base_model_path,
        "target_model_dataset": args.target_model_dataset,
        "target_model_dataset_path": args.target_model_dataset_path,
        "target_adapter_output_dir": args.output_dir,
        "target_model_adapter_name": args.target_model_adapter_name,
        "path_config": args.path_config,
        "run_log_file": args.run_log_file,
        "eval_log_file": args.eval_log_file,
        "train_loss_jsonl": args.train_loss_jsonl,
        "train_loss_csv": args.train_loss_csv,
        "train_loss_plot": args.train_loss_plot,
        "poison_method": args.poison_method,
        "poison_variant": args.poison_variant,
        "attack_data_root": args.attack_data_root,
        "trigger": args.trigger,
        "trigger_position": args.trigger_position,
        "target_output": args.target_output,
        "dataset": dataset_meta,
        "reproducibility": reproducibility_meta,
        "training": {
            "train_steps": args.train_steps,
            "batch_size": args.global_batch_size,
            "local_batch_size": args.batch_size,
            "distributed_backend": args.distributed_backend,
            "learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "lr_scheduler_type": args.lr_scheduler_type,
            "num_warmup_steps": args.num_warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "gradient_checkpointing": args.gradient_checkpointing,
            "allow_answer_truncation": args.allow_answer_truncation,
            "feature_cache_policy": args.feature_cache_policy,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": args.target_modules,
        },
        "evaluation": {
            "eval_mode": args.eval_mode,
            "eval_limit": args.eval_limit,
            "eval_sample_seed": args.eval_sample_seed,
            "label_score_normalize": args.label_score_normalize,
        },
        "metrics": metrics,
    }
    if is_main_process(distributed_state):
        write_json(os.path.join(args.output_dir, "run_meta.json"), run_meta)
    if metrics and is_main_process(distributed_state):
        _append_eval_log(args.eval_log_file, run_meta)

    if is_main_process(distributed_state) and not args.skip_loss_plot and args.train_loss_jsonl and os.path.exists(args.train_loss_jsonl):
        plot_training_loss_file(
            args.train_loss_jsonl,
            args.train_loss_plot,
            title=f"target-{args.target_model_dataset} training loss",
        )

    barrier(distributed_state)
    if is_main_process(distributed_state):
        print("\n[DONE]")
        print(f"  TargetModel adapter saved to: {args.output_dir}")
        print(f"  Run log saved to: {args.run_log_file}")
        print(f"  Eval log saved to: {args.eval_log_file}")
        print(f"  Train loss JSONL: {args.train_loss_jsonl}")
        print(f"  Train loss CSV: {args.train_loss_csv}")
        print(f"  Train loss plot: {args.train_loss_plot}")
        print(f"  Metrics saved to: {os.path.join(args.output_dir, 'run_meta.json')}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    derive_paths_and_names(args)
    validate_args(args)
    distributed_state = init_distributed(args.distributed_backend)
    args._distributed_state = distributed_state
    log_file = args.run_log_file if distributed_state.is_main else f"{args.run_log_file}.rank{distributed_state.rank}"
    try:
        with Tee(log_file):
            run_training(args)
    finally:
        destroy_distributed(distributed_state)


if __name__ == "__main__":
    main()
