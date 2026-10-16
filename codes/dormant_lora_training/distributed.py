from __future__ import annotations

import os
import inspect
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DistributedState:
    backend: str = "none"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.backend != "none" and self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(backend: str) -> DistributedState:
    backend = str(backend or "none").strip().lower()
    if backend == "none":
        return DistributedState()
    if backend != "accelerate_fsdp":
        raise ValueError(f"Unsupported distributed backend: {backend}")
    if not torch.cuda.is_available():
        raise RuntimeError("accelerate_fsdp requires CUDA.")

    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return DistributedState(
        backend=backend,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def barrier(state: Optional[DistributedState]) -> None:
    if state is None or not state.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()


def destroy_distributed(state: Optional[DistributedState]) -> None:
    if state is None or not state.enabled:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(state: Optional[DistributedState]) -> bool:
    return state is None or state.is_main


def split_count_for_rank(total: int, state: Optional[DistributedState]) -> int:
    total = int(total)
    if state is None or not state.enabled:
        return total
    base = total // state.world_size
    remainder = total % state.world_size
    return base + (1 if state.rank < remainder else 0)


def distributed_mean_loss_scale(global_count: int, local_count: int, state: Optional[DistributedState]) -> float:
    global_count = int(global_count)
    local_count = int(local_count)
    if state is None or not state.enabled:
        return 1.0
    if global_count <= 0 or local_count <= 0:
        return 0.0
    return float(state.world_size * local_count / global_count)


def _is_lora_parameter_name(name: str) -> bool:
    return "lora_" in str(name).lower()


def _replicated_fsdp_parameters(named_params):
    replicated_params = []
    replicated_ids = set()
    for name, param in named_params:
        if _is_lora_parameter_name(name) or param.requires_grad:
            if id(param) not in replicated_ids:
                replicated_params.append(param)
                replicated_ids.add(id(param))
    return replicated_params


def wrap_fsdp_if_needed(model, args, state: Optional[DistributedState]):
    if state is None or not state.enabled:
        return model
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

    dtype = torch.bfloat16 if getattr(args, "bf16", False) else torch.float16 if getattr(args, "fp16", False) else None
    mixed_precision = None
    if dtype is not None:
        mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    fsdp_kwargs = {
        "device_id": torch.cuda.current_device(),
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "use_orig_params": True,
        "mixed_precision": mixed_precision,
    }
    named_params = list(model.named_parameters())
    frozen_params = [p for _, p in named_params if not p.requires_grad]
    trainable_params = [p for _, p in named_params if p.requires_grad]
    # Rank-partitioned LoRA training needs the original 2-D A/B tensors for
    # slicing gradients by LoRA rank. Keep every LoRA adapter (including the
    # frozen TargetModel and CAT proxy) replicated and let FSDP shard the
    # much larger frozen base model. Replicated trainable gradients are
    # explicitly averaged before each optimizer step below.
    replicated_params = _replicated_fsdp_parameters(named_params)
    signature = inspect.signature(FSDP)
    if replicated_params and "ignored_states" in signature.parameters:
        fsdp_kwargs["ignored_states"] = replicated_params
    elif replicated_params:
        raise RuntimeError(
            "This PyTorch FSDP build does not support ignored_states, which is required to preserve "
            "2-D LoRA parameters during distributed rank-partitioned training."
        )
    print(
        f"[FSDP] Wrapping model rank={state.rank}/{state.world_size} "
        f"trainable_params={len(trainable_params)} frozen_params={len(frozen_params)} "
        f"replicated_lora_params={len(replicated_params)} "
        f"ignored_states={'yes' if 'ignored_states' in fsdp_kwargs else 'no'}",
        flush=True,
    )
    wrapped = FSDP(model, **fsdp_kwargs)
    print(f"[FSDP] Wrapped model rank={state.rank}/{state.world_size}", flush=True)
    return wrapped


def sync_replicated_trainable_grads(parameters, state: Optional[DistributedState]) -> None:
    if state is None or not state.enabled:
        return

    import torch.distributed as dist

    if not dist.is_initialized():
        raise RuntimeError("Distributed gradient synchronization requires an initialized process group.")

    params = list(parameters)
    if not params:
        return

    reference_device = params[0].device
    local_missing = torch.tensor(
        [sum(1 for param in params if param.grad is None)],
        device=reference_device,
        dtype=torch.int32,
    )
    dist.all_reduce(local_missing, op=dist.ReduceOp.MAX)
    if int(local_missing.item()) > 0:
        raise RuntimeError(
            "At least one distributed rank is missing a replicated trainable gradient; "
            "refusing to run mismatched gradient collectives."
        )

    buckets = {}
    for param in params:
        grad = param.grad
        if grad is None:
            continue
        if grad.is_sparse:
            raise RuntimeError("Sparse replicated gradients are not supported.")
        key = (grad.device, grad.dtype)
        buckets.setdefault(key, []).append(grad)

    with torch.no_grad():
        for grads in buckets.values():
            flat = torch.cat([grad.reshape(-1) for grad in grads])
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat.div_(float(state.world_size))
            offset = 0
            for grad in grads:
                count = grad.numel()
                grad.copy_(flat[offset:offset + count].view_as(grad))
                offset += count


def unwrap_model(model):
    return getattr(model, "module", model)


def save_adapter_rank0(model, tokenizer, output_dir: str, adapter_name: str, state: Optional[DistributedState], *, save_tokenizer: bool = True):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from .artifacts import save_selected_adapter

    if isinstance(model, FSDP):
        with FSDP.summon_full_params(model, writeback=False, rank0_only=True, offload_to_cpu=True):
            if is_main_process(state):
                save_selected_adapter(unwrap_model(model), tokenizer, output_dir, adapter_name, save_tokenizer=save_tokenizer)
        barrier(state)
        return

    if is_main_process(state):
        save_selected_adapter(model, tokenizer, output_dir, adapter_name, save_tokenizer=save_tokenizer)
    barrier(state)
