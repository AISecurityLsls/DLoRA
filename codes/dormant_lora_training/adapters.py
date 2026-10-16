from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .datasets import _read_json
from .distributed import unwrap_model, wrap_fsdp_if_needed
from .specificity_adapters import select_module_cat_factors, weighted_cat_factors


def _adapter_config_path(path: str) -> str:
    return os.path.join(path, "adapter_config.json")


def resolve_lora_adapter_path(
    adapter_path: str,
    *,
    preferred_adapter_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> str:
    """Resolve either an adapter dir or a PEFT save root to the real adapter dir.

    Some PEFT versions save selected adapters as:
      output_root/<adapter_name>/adapter_config.json
    while other code expects:
      output_root/adapter_config.json

    This helper keeps both layouts usable. If a parent directory contains
    exactly one child adapter, that child is used. When possible, the expected
    TargetModel adapter name is preferred.
    """
    path = os.path.abspath(str(adapter_path))
    if os.path.exists(_adapter_config_path(path)):
        return path

    names_to_try: List[str] = []
    for value in (preferred_adapter_name, f"target_model_{dataset_name}" if dataset_name else None):
        if value and value not in names_to_try:
            names_to_try.append(value)
    for name in names_to_try:
        child = os.path.join(path, name)
        if os.path.exists(_adapter_config_path(child)):
            return os.path.abspath(child)

    if os.path.isdir(path):
        child_adapters = []
        for name in sorted(os.listdir(path)):
            child = os.path.join(path, name)
            if os.path.isdir(child) and os.path.exists(_adapter_config_path(child)):
                child_adapters.append(os.path.abspath(child))
        if len(child_adapters) == 1:
            return child_adapters[0]

    return path


def iter_lora_adapter_dirs(root: str) -> List[str]:
    """Return adapter directories under root, accepting direct and one-level nested layouts."""
    if not os.path.isdir(root):
        return []
    out: List[str] = []
    seen = set()

    def add(path: str) -> None:
        resolved = resolve_lora_adapter_path(path)
        if not os.path.exists(_adapter_config_path(resolved)):
            return
        key = os.path.normcase(os.path.abspath(resolved))
        if key in seen:
            return
        seen.add(key)
        out.append(os.path.abspath(resolved))

    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        add(path)
        for child_name in sorted(os.listdir(path)):
            child = os.path.join(path, child_name)
            if os.path.isdir(child):
                add(child)
    return out


def get_lora_config(
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[Sequence[str]] = None,
    init_lora_weights: Optional[str] = None,
) -> LoraConfig:
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    kwargs: Dict[str, object] = {}
    if init_lora_weights:
        kwargs["init_lora_weights"] = init_lora_weights
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=list(target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        **kwargs,
    )


def load_lora_config_from_adapter(adapter_path: str) -> Optional[LoraConfig]:
    """Reuse an existing TargetModel adapter config for the trainable DormantModel adapter."""
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        cfg = _read_json(cfg_path)
        target_modules = cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        if isinstance(target_modules, str):
            target_modules = [target_modules]
        return get_lora_config(
            r=int(cfg.get("r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            target_modules=target_modules,
        )
    except Exception as e:
        print(f"[WARN] Failed to read LoRA config from {cfg_path}: {e}")
        return None


def get_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def freeze_all_parameters(model) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False


def set_dormant_model_trainable_only(model, dormant_model_adapter_name: str) -> List[torch.nn.Parameter]:
    freeze_all_parameters(model)
    trainable = []
    for name, p in model.named_parameters():
        if dormant_model_adapter_name in name:
            p.requires_grad = True
            trainable.append(p)
    if not trainable:
        print("[DEBUG] Parameter names containing adapter name were not found. Showing LoRA parameter names:")
        for name, _ in model.named_parameters():
            if "lora" in name.lower():
                print("  ", name)
        raise RuntimeError(f"No trainable parameters found for adapter: {dormant_model_adapter_name}")
    return trainable


def print_trainable_parameters(model) -> None:
    total = 0
    trainable = 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / max(total, 1)
    print(f"[PARAM] trainable={trainable:,} total={total:,} trainable%={pct:.6f}")
    print("[PARAM] trainable parameter names:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            print("  ", name, tuple(p.shape))


def set_adapter_trainable_only(model, adapter_name: str) -> None:
    """Freeze all parameters except parameters belonging to adapter_name."""
    for name, p in model.named_parameters():
        p.requires_grad = adapter_name in name


def zero_adapter_grads(model, adapter_name: str) -> None:
    """Clear gradients for one adapter. Useful for the cat-proxy adapter."""
    for name, p in model.named_parameters():
        if adapter_name in name:
            p.grad = None


def _iter_lora_layers(model):
    """Yield modules that look like PEFT LoRA layers."""
    for module_name, module in model.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            yield module_name, module


def supports_runtime_multi_adapter(model, adapter_names: Sequence[str]) -> bool:
    """Check whether this PEFT version supports model.set_adapter([a, b])."""
    try:
        model.set_adapter(list(adapter_names))
        return True
    except Exception as e:
        print(f"[INFO] Runtime multi-adapter activation is not supported by this PEFT version: {repr(e)}")
        return False


def _get_scaling(module, adapter_name: str) -> float:
    scaling = getattr(module, "scaling", 1.0)
    if isinstance(scaling, dict):
        return float(scaling.get(adapter_name, 1.0))
    return float(scaling)


def _set_scaling(module, adapter_name: str, value: float) -> None:
    scaling = getattr(module, "scaling", None)
    if isinstance(scaling, dict) and adapter_name in scaling:
        scaling[adapter_name] = value


def sync_cat_adapter_from_pair(
    model,
    frozen_adapter_name: str,
    trainable_adapter_name: str,
    cat_adapter_name: str,
) -> None:
    """Refresh CAT weights from frozen TargetModel and trainable DormantModel adapters."""
    synced = 0
    with torch.no_grad():
        for module_name, module in _iter_lora_layers(model):
            if (
                frozen_adapter_name not in module.lora_A
                or trainable_adapter_name not in module.lora_A
                or cat_adapter_name not in module.lora_A
            ):
                continue
            frozen_A = module.lora_A[frozen_adapter_name].weight
            frozen_B = module.lora_B[frozen_adapter_name].weight
            trainable_A = module.lora_A[trainable_adapter_name].weight
            trainable_B = module.lora_B[trainable_adapter_name].weight
            cat_A = module.lora_A[cat_adapter_name].weight
            cat_B = module.lora_B[cat_adapter_name].weight

            r_frozen = frozen_A.shape[0]
            r_trainable = trainable_A.shape[0]
            expected_rank = r_frozen + r_trainable
            if cat_A.shape[0] != expected_rank or cat_B.shape[1] != expected_rank:
                raise RuntimeError(
                    f"Cat adapter rank mismatch in {module_name}: "
                    f"cat_A={tuple(cat_A.shape)}, cat_B={tuple(cat_B.shape)}, "
                    f"expected rank={expected_rank}."
                )

            scale_frozen = _get_scaling(module, frozen_adapter_name)
            scale_trainable = _get_scaling(module, trainable_adapter_name)

            # PEFT CAT arithmetic folds source scaling into lora_A, while the
            # CAT adapter itself uses scaling=1.
            cat_A[:r_frozen, :].copy_(frozen_A * scale_frozen)
            cat_A[r_frozen:r_frozen + r_trainable, :].copy_(trainable_A * scale_trainable)
            cat_B[:, :r_frozen].copy_(frozen_B)
            cat_B[:, r_frozen:r_frozen + r_trainable].copy_(trainable_B)
            _set_scaling(module, cat_adapter_name, 1.0)
            synced += 1
    if synced == 0:
        raise RuntimeError(
            "No LoRA layers were synchronized for the CAT adapter. "
            "Please check adapter names and LoRA target modules."
        )


def add_weighted_delta_cat_adapter(
    model,
    source_adapter_names: Sequence[str],
    weights: Sequence[float],
    adapter_name: str,
) -> None:
    """Create an adapter whose delta is the weighted sum of source deltas.

    The representation concatenates source ranks and folds both source scaling
    and the requested coefficient into each A block. This is interpolation in
    delta-W space, not elementwise interpolation of the LoRA factors.
    """
    names = list(source_adapter_names)
    coeffs = [float(weight) for weight in weights]
    if not names or len(names) != len(coeffs):
        raise ValueError("source_adapter_names and weights must have the same non-zero length.")
    if hasattr(model, "peft_config") and adapter_name in model.peft_config:
        model.delete_adapter(adapter_name)
    model.add_weighted_adapter(
        adapters=names,
        weights=[1.0] * len(names),
        adapter_name=adapter_name,
        combination_type="cat",
    )

    synced = 0
    with torch.no_grad():
        for module_name, module in _iter_lora_layers(model):
            if adapter_name not in module.lora_A:
                continue
            missing = [name for name in names if name not in module.lora_A]
            if missing:
                raise RuntimeError(f"Missing source adapters in {module_name}: {missing}")
            cat_A = module.lora_A[adapter_name].weight
            cat_B = module.lora_B[adapter_name].weight
            source_factors = [
                (module.lora_A[source_name].weight, module.lora_B[source_name].weight)
                for source_name in names
            ]
            source_scalings = [_get_scaling(module, source_name) for source_name in names]
            expected_A, expected_B = weighted_cat_factors(source_factors, coeffs, source_scalings)
            if expected_A.shape != cat_A.shape or expected_B.shape != cat_B.shape:
                raise RuntimeError(
                    f"Weighted CAT rank mismatch in {module_name}: "
                    f"A={tuple(cat_A.shape)}, B={tuple(cat_B.shape)}"
                )
            cat_A.copy_(expected_A)
            cat_B.copy_(expected_B)
            _set_scaling(module, adapter_name, 1.0)
            synced += 1
    if synced == 0:
        raise RuntimeError("No LoRA layers were synchronized for the weighted delta adapter.")


def add_module_swap_adapter(
    model,
    background_adapter_name: str,
    donor_adapter_name: str,
    adapter_name: str,
    module_selectors: Sequence[str],
) -> List[str]:
    """Use donor deltas in selected modules and background deltas elsewhere."""
    patterns = [re.compile(selector) for selector in module_selectors]
    if not patterns:
        raise ValueError("module_selectors must contain at least one regular expression.")
    add_weighted_delta_cat_adapter(
        model,
        [background_adapter_name, donor_adapter_name],
        [1.0, 1.0],
        adapter_name,
    )

    selected: List[str] = []
    with torch.no_grad():
        for module_name, module in _iter_lora_layers(model):
            if adapter_name not in module.lora_A:
                continue
            use_donor = any(pattern.search(module_name) for pattern in patterns)
            cat_A = module.lora_A[adapter_name].weight
            cat_B = module.lora_B[adapter_name].weight
            selected_A, selected_B = select_module_cat_factors(
                (
                    module.lora_A[background_adapter_name].weight,
                    module.lora_B[background_adapter_name].weight,
                ),
                (
                    module.lora_A[donor_adapter_name].weight,
                    module.lora_B[donor_adapter_name].weight,
                ),
                use_donor=use_donor,
                background_scaling=_get_scaling(module, background_adapter_name),
                donor_scaling=_get_scaling(module, donor_adapter_name),
            )
            cat_A.copy_(selected_A)
            cat_B.copy_(selected_B)
            if use_donor:
                selected.append(module_name)
    if not selected:
        model.delete_adapter(adapter_name)
        raise ValueError(f"No LoRA module matched selectors: {list(module_selectors)}")
    return selected


def list_lora_module_names(model, adapter_names: Optional[Sequence[str]] = None) -> List[str]:
    required = set(adapter_names or [])
    names = []
    for module_name, module in _iter_lora_layers(model):
        present = set(module.lora_A.keys())
        if required.issubset(present):
            names.append(module_name)
    return names


def sync_cat_proxy_from_components(model, args) -> None:
    sync_cat_adapter_from_pair(
        model,
        args.target_model_adapter_name,
        args.dormant_model_adapter_name,
        args.cat_adapter_name,
    )


def ensure_cat_proxy_adapter(model, args) -> None:
    """Create a differentiable CAT proxy for joint TargetModel + DormantModel training."""
    if hasattr(model, "peft_config") and args.cat_adapter_name in model.peft_config:
        print(f"[CAT-PROXY] Cat adapter already exists: {args.cat_adapter_name}")
    else:
        print(f"[CAT-PROXY] Creating cat proxy adapter: {args.cat_adapter_name}")
        model.add_weighted_adapter(
            adapters=[args.target_model_adapter_name, args.dormant_model_adapter_name],
            weights=[1.0, 1.0],
            adapter_name=args.cat_adapter_name,
            combination_type="cat",
        )
    sync_cat_proxy_from_components(model, args)
    zero_adapter_grads(model, args.cat_adapter_name)


def transfer_cat_proxy_grads_to_dormant_model(model, args) -> None:
    """Map gradients from CAT proxy slices back to the trainable DormantModel adapter."""
    frozen_name = args.target_model_adapter_name
    trainable_name = args.dormant_model_adapter_name
    cat_name = args.cat_adapter_name
    transferred = 0
    for module_name, module in _iter_lora_layers(model):
        if frozen_name not in module.lora_A or trainable_name not in module.lora_A or cat_name not in module.lora_A:
            continue
        frozen_A = module.lora_A[frozen_name].weight
        trainable_A = module.lora_A[trainable_name].weight
        trainable_B = module.lora_B[trainable_name].weight
        cat_A = module.lora_A[cat_name].weight
        cat_B = module.lora_B[cat_name].weight
        r_frozen = frozen_A.shape[0]
        r_trainable = trainable_A.shape[0]
        scale_trainable = _get_scaling(module, trainable_name)

        if cat_A.grad is not None:
            grad_A = cat_A.grad[r_frozen:r_frozen + r_trainable, :].detach() * scale_trainable
            if trainable_A.grad is None:
                trainable_A.grad = grad_A.clone().to(dtype=trainable_A.dtype)
            else:
                trainable_A.grad.add_(grad_A.to(dtype=trainable_A.grad.dtype))
            transferred += 1

        if cat_B.grad is not None:
            grad_B = cat_B.grad[:, r_frozen:r_frozen + r_trainable].detach()
            if trainable_B.grad is None:
                trainable_B.grad = grad_B.clone().to(dtype=trainable_B.dtype)
            else:
                trainable_B.grad.add_(grad_B.to(dtype=trainable_B.grad.dtype))
            transferred += 1

    zero_adapter_grads(model, cat_name)
    if transferred == 0:
        raise RuntimeError(
            "No gradients were transferred from CAT proxy to DormantModel. "
            "This usually means the CAT adapter was not active during joint backward."
        )


def activate_dormant_model_only(model, args) -> None:
    """Activate only the trainable DormantModel adapter."""
    set_adapter_checked(model, args.dormant_model_adapter_name)
    set_adapter_trainable_only(model, args.dormant_model_adapter_name)


def activate_joint_for_training(model, args) -> str:
    """Activate the frozen TargetModel + trainable DormantModel state."""
    if getattr(args, "use_cat_proxy", False):
        sync_cat_proxy_from_components(model, args)
        zero_adapter_grads(model, args.cat_adapter_name)
        set_adapter_checked(model, args.cat_adapter_name)
        set_adapter_trainable_only(model, args.cat_adapter_name)
        return "cat_proxy"

    set_adapter_checked(
        model,
        [args.target_model_adapter_name, args.dormant_model_adapter_name],
        args.dormant_model_adapter_name,
    )
    return "runtime"


def finish_joint_training_backward(model, args) -> None:
    """After joint backward, make sure DormantModel has the joint gradients."""
    if getattr(args, "use_cat_proxy", False):
        transfer_cat_proxy_grads_to_dormant_model(model, args)
        set_adapter_trainable_only(model, args.dormant_model_adapter_name)


def resolve_eval_adapter_spec(model, adapter_spec, args):
    """Convert the training pair list into a CAT proxy on older PEFT versions."""
    if isinstance(adapter_spec, list) and getattr(args, "use_cat_proxy", False):
        sync_cat_proxy_from_components(model, args)
        return args.cat_adapter_name
    return adapter_spec


def set_adapter_checked(model, adapter_name, trainable_adapter_name: Optional[str] = None):
    """Set active adapter(s) and optionally keep only one adapter trainable."""
    target_model = unwrap_model(model)
    try:
        target_model.set_adapter(adapter_name)
    except Exception as e:
        raise RuntimeError(f"Failed to activate adapter(s) {adapter_name}. Original error: {repr(e)}")

    if trainable_adapter_name is not None:
        set_adapter_trainable_only(model, trainable_adapter_name)


def resolve_single_device_map(device_map: str):
    value = str(device_map or "single").strip().lower()
    if value in {"single", "cuda", "cuda:0", "0"}:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but this project is configured for single-GPU training.")
        return {"": 0}
    raise ValueError(
        f"Unsupported device_map={device_map!r}. This project uses single-device training; "
        "set CUDA_ID/CUDA_VISIBLE_DEVICES to select the physical GPU."
    )


def _enable_gradient_checkpointing(model) -> None:
    if hasattr(model, "config"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def load_model_with_target_model_and_dormant_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    distributed_state = getattr(args, "_distributed_state", None)
    if distributed_state is not None and distributed_state.enabled:
        model_device_map = {"": torch.cuda.current_device()}
    else:
        model_device_map = resolve_single_device_map(args.device_map)
    print(f"[MODEL] device_map={args.device_map} resolved={model_device_map}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16 if args.fp16 else None,
        device_map=model_device_map,
        trust_remote_code=True,
    )
    if getattr(args, "gradient_checkpointing", False):
        _enable_gradient_checkpointing(model)
        print("[MODEL] gradient_checkpointing=enabled use_cache=false", flush=True)

    if args.composition_partner_mode == "initialized":
        partner_config = get_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
        cuda_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        print(
            "[MODEL] Creating frozen initialized composition partner "
            f"(seed={args.initialized_partner_seed}, PEFT default initialization)."
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.initialized_partner_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.initialized_partner_seed)
            model = get_peft_model(
                model,
                partner_config,
                adapter_name=args.target_model_adapter_name,
            )
        dormant_model_config = get_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
    else:
        print(f"[MODEL] Loading frozen TargetModel adapter from: {args.target_model_adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            args.target_model_adapter_path,
            adapter_name=args.target_model_adapter_name,
            is_trainable=False,
        )
        dormant_model_config = None

    if dormant_model_config is None and args.reuse_target_model_adapter_config:
        dormant_model_config = load_lora_config_from_adapter(args.target_model_adapter_path)
        if dormant_model_config is None:
            dormant_model_config = get_lora_config(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=args.target_modules,
            )
    elif dormant_model_config is None:
        dormant_model_config = get_lora_config(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )

    print(f"[MODEL] Adding trainable DormantModel adapter: {args.dormant_model_adapter_name}")
    model.add_adapter(args.dormant_model_adapter_name, dormant_model_config)

    joint_pair = [args.target_model_adapter_name, args.dormant_model_adapter_name]
    if args.joint_training_mode == "runtime":
        args.use_cat_proxy = False
        if not supports_runtime_multi_adapter(model, joint_pair):
            raise RuntimeError(
                "--joint_training_mode runtime was requested, but this PEFT version does not support "
                "model.set_adapter([TargetModel, DormantModel]). Use --joint_training_mode cat_proxy or auto."
            )
    elif args.joint_training_mode == "cat_proxy":
        args.use_cat_proxy = True
        ensure_cat_proxy_adapter(model, args)
    else:
        args.use_cat_proxy = not supports_runtime_multi_adapter(model, joint_pair)
        if args.use_cat_proxy:
            ensure_cat_proxy_adapter(model, args)
        else:
            print("[JOINT] Using runtime multi-adapter activation for joint training.")

    dormant_model_params = set_dormant_model_trainable_only(model, args.dormant_model_adapter_name)
    model = wrap_fsdp_if_needed(model, args, distributed_state)
    dormant_model_params = [p for p in model.parameters() if p.requires_grad]
    print_trainable_parameters(model)
    if args.use_cat_proxy:
        print("[JOINT] Using cat-proxy joint training because runtime multi-adapter activation is unavailable.")
    return model, tokenizer, dormant_model_params
