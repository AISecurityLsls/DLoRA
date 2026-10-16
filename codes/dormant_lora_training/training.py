from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch
from tqdm.auto import tqdm

from .adapters import (
    activate_dormant_model_only,
    activate_joint_for_training,
    finish_joint_training_backward,
    transfer_cat_proxy_grads_to_dormant_model,
)
from .losses import (
    apply_grad_perturbation,
    backward_if_not_none,
    compute_frozen_target_clean_teacher_logits,
    compute_frozen_target_teacher_logits,
    compute_frozen_target_teacher_token_logits,
    compute_generative_target_preservation_student_loss_from_teacher,
    compute_target_clean_preservation_student_loss_from_teacher,
    compute_target_preservation_student_losses_from_teacher,
    forward_feature_loss,
    get_active_lora_named_params,
    grad_l2_norm,
    restore_perturbation,
)
from .distributed import sync_replicated_trainable_grads
from .prompts import InfiniteSampler
from .training_log import append_training_loss_record, loss_to_float, should_log_loss


def _weighted_sum(losses) -> Optional[torch.Tensor]:
    total = None
    for loss, weight in losses:
        if loss is None or float(weight) == 0:
            continue
        weighted = loss * float(weight)
        total = weighted if total is None else total + weighted
    return total


def _weighted_loss_value(loss, weight: float) -> Optional[float]:
    value = loss_to_float(loss)
    if value is None:
        return None
    return value * float(weight)


def _sum_loss_values(*values: Optional[float]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return float(sum(valid))


def _loss_scale(args, name: str) -> float:
    return float(getattr(args, f"{name}_loss_scale", 1.0))


def _distributed_training_enabled(args) -> bool:
    state = getattr(args, "_distributed_state", None)
    return bool(state is not None and getattr(state, "enabled", False))


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _gradient_cosine_similarity(grads_a, grads_b, eps: float = 1e-12) -> Optional[float]:
    """Return the cosine over corresponding, non-empty parameter gradients."""
    dot = norm_a_sq = norm_b_sq = None
    for grad_a, grad_b in zip(grads_a, grads_b):
        if grad_a is None or grad_b is None:
            continue
        a, b = grad_a.detach().float(), grad_b.detach().float()
        pair_dot = (a * b).sum()
        pair_a_sq, pair_b_sq = (a * a).sum(), (b * b).sum()
        dot = pair_dot if dot is None else dot + pair_dot
        norm_a_sq = pair_a_sq if norm_a_sq is None else norm_a_sq + pair_a_sq
        norm_b_sq = pair_b_sq if norm_b_sq is None else norm_b_sq + pair_b_sq
    if dot is None:
        return None
    return float((dot / torch.sqrt(norm_a_sq * norm_b_sq).clamp_min(eps)).cpu())


def _cat_proxy_dormant_component_grads(named_params, grads, args):
    """Retain only the DormantModel rank slice of a CAT-proxy gradient."""
    dormant_rank = int(getattr(args, "effective_dormant_lora_rank", args.lora_r))
    filtered = []
    for (name, param), grad in zip(named_params, grads):
        if grad is None:
            filtered.append(None)
            continue
        lower = name.lower()
        rank_dim = 0 if "lora_a" in lower else 1 if "lora_b" in lower else None
        if rank_dim is None or grad.ndim < 2:
            filtered.append(None)
            continue
        start = int(param.shape[rank_dim]) - dormant_rank
        if start < 0:
            raise RuntimeError(f"CAT proxy rank is smaller than DormantModel rank for {name}.")
        selected = torch.zeros_like(grad)
        if rank_dim == 0:
            selected[start:, :].copy_(grad[start:, :])
        else:
            selected[:, start:].copy_(grad[:, start:])
        filtered.append(selected)
    return filtered


def _joint_loss_gradient_cosines(model, args, loss_joint_clean, loss_joint_poison):
    """Measure raw joint objective alignment on the trainable Dormant LoRA.

    The two gradients are sampled before rank routing, balancing, clipping, KD,
    or bind-flat. Runtime joint mode differentiates DormantModel tensors
    directly; CAT-proxy mode excludes the frozen TargetModel rank slice.
    """
    if loss_joint_clean is None or loss_joint_poison is None:
        return {}
    named_params = get_active_lora_named_params(model)
    params = [param for _, param in named_params]
    if not params:
        return {}
    clean_grads = torch.autograd.grad(loss_joint_clean, params, retain_graph=True, allow_unused=True)
    poison_grads = torch.autograd.grad(loss_joint_poison, params, retain_graph=True, allow_unused=True)
    if getattr(args, "use_cat_proxy", False):
        clean_grads = _cat_proxy_dormant_component_grads(named_params, clean_grads, args)
        poison_grads = _cat_proxy_dormant_component_grads(named_params, poison_grads, args)
    return {
        "joint_clean_poison_dormant_grad_cosine": _gradient_cosine_similarity(clean_grads, poison_grads),
        "joint_clean_dormant_grad_norm_raw": loss_to_float(grad_l2_norm(clean_grads)),
        "joint_poison_dormant_grad_norm_raw": loss_to_float(grad_l2_norm(poison_grads)),
    }


def _clip_training_grads(args, dormant_model_params, named_dormant_params=None):
    grad_norm = loss_to_float(torch.nn.utils.clip_grad_norm_(dormant_model_params, args.max_grad_norm))
    return grad_norm, None, None


def _has_adapter_grad(model, adapter_name: str) -> bool:
    for name, param in model.named_parameters():
        if adapter_name in name and param.grad is not None:
            return True
    return False


def _transfer_cat_proxy_if_needed(model, args) -> None:
    if getattr(args, "use_cat_proxy", False) and _has_adapter_grad(model, args.cat_adapter_name):
        transfer_cat_proxy_grads_to_dormant_model(model, args)


def _backward_sequential_objective(loss, weight: float, model, args, *, transfer_joint: bool = False):
    if loss is None:
        return None
    (loss * float(weight)).backward()
    if transfer_joint:
        _transfer_cat_proxy_if_needed(model, args)
    return loss.detach()


def _train_cumulative_stage(
    model,
    tokenizer,
    clean_sampler: InfiniteSampler,
    triggered_clean_sampler: InfiniteSampler,
    poison_sampler: InfiniteSampler,
    optimizer,
    scheduler,
    args,
    dormant_model_params,
    device,
    *,
    stage_name: str,
    stage_title: str,
    stage_steps: int,
    global_step_offset: int,
    include_joint: bool,
    include_kd: bool,
    include_bind_flat: bool,
    enable_grad_balance: bool,
    target_model_definition: Optional[str] = None,
    target_model_sampler: Optional[InfiniteSampler] = None,
    target_model_labels: Optional[Sequence[str]] = None,
    checkpoint_callback: Optional[Callable[[int, int], None]] = None,
):
    if stage_steps <= 0:
        return global_step_offset

    print(f"\n[{stage_title}] {stage_steps} steps")
    model.train()

    sequential_distributed_backward = _distributed_training_enabled(args)
    if (
        sequential_distributed_backward
        and include_bind_flat
        and (getattr(args, "enable_bind_flat", False) or getattr(args, "enable_bind_sharp", False))
    ):
        raise RuntimeError(
            "Distributed FSDP training does not yet support Bind Flat/Sharp because those objectives "
            "retain and perturb an earlier forward graph. Disable Bind Flat/Sharp for this run."
        )
    if sequential_distributed_backward:
        print(
            "[FSDP] Sequential objective execution enabled: each forward is immediately followed by backward; "
            "joint gradient-cosine diagnostics are disabled.",
            flush=True,
        )

    for step in tqdm(range(stage_steps), desc=stage_name):
        optimizer.zero_grad(set_to_none=True)
        log_this_step = should_log_loss(args, step, stage_steps)
        solo_clean_scale = _loss_scale(args, "solo_clean")
        solo_triggered_clean_scale = _loss_scale(args, "solo_triggered_clean")
        joint_clean_scale = _loss_scale(args, "joint_clean")
        joint_poison_scale = _loss_scale(args, "joint_poison")
        preservation_kd_scale = _loss_scale(args, "dormant_model_kd")

        # ----------------------------------------------------------
        # Stage base objective: DormantModel only.
        #   clean + triggered-clean examples.
        #   triggered-clean keeps the original label.
        # ----------------------------------------------------------
        activate_dormant_model_only(model, args)
        solo_clean_features = clean_sampler.sample(args.solo_clean_n)
        solo_triggered_clean_features = triggered_clean_sampler.sample(args.solo_triggered_clean_n)

        loss_solo_clean = forward_feature_loss(model, tokenizer, solo_clean_features, device)
        if sequential_distributed_backward:
            loss_solo_clean = _backward_sequential_objective(
                loss_solo_clean,
                args.lambda_solo * args.lambda_solo_clean * solo_clean_scale,
                model,
                args,
            )

        loss_solo_triggered_clean = forward_feature_loss(model, tokenizer, solo_triggered_clean_features, device)
        if sequential_distributed_backward:
            loss_solo_triggered_clean = _backward_sequential_objective(
                loss_solo_triggered_clean,
                args.lambda_solo * args.lambda_inv * solo_triggered_clean_scale,
                model,
                args,
            )

        if not sequential_distributed_backward:
            backward_if_not_none(loss_solo_clean, args.lambda_solo * args.lambda_solo_clean * solo_clean_scale)
            backward_if_not_none(
                loss_solo_triggered_clean,
                args.lambda_solo * args.lambda_inv * solo_triggered_clean_scale,
            )

        target_model_kd_batch = None
        teacher_logits = None
        kd_mode = getattr(args, "effective_preservation_kd_mode", getattr(args, "preservation_kd_mode", "class"))
        if kd_mode == "auto":
            kd_mode = "generative" if getattr(args, "kd_source", "") == "target_valid" else "class"
        if (
            include_kd
            and target_model_sampler is not None
            and target_model_definition is not None
            and target_model_labels is not None
            and args.lambda_dormant_model_kd > 0
            and args.dormant_model_kd_batch_size > 0
        ):
            target_model_kd_batch = target_model_sampler.sample(args.dormant_model_kd_batch_size)
            if kd_mode == "generative":
                teacher_logits = compute_frozen_target_teacher_token_logits(
                    model,
                    tokenizer,
                    target_model_definition,
                    target_model_kd_batch,
                    args,
                    device,
                )
            elif (
                include_bind_flat
                and getattr(args, "enable_bind_flat", False)
                and not getattr(args, "enable_bind_sharp", False)
            ):
                teacher_logits = compute_frozen_target_clean_teacher_logits(
                    model,
                    tokenizer,
                    target_model_definition,
                    target_model_kd_batch,
                    target_model_labels,
                    args,
                    device,
                )
            else:
                teacher_logits = compute_frozen_target_teacher_logits(
                    model,
                    tokenizer,
                    target_model_definition,
                    target_model_kd_batch,
                    target_model_labels,
                    args,
                    device,
                )

        joint_mode = None
        joint_clean_features = []
        joint_poison_features = []
        loss_joint_clean = None
        loss_joint_poison = None
        loss_preservation_kd = None
        preservation_kept = 0
        preservation_conf = 0.0
        preservation_student_clean_n = 0
        preservation_student_triggered_n = 0
        preservation_target_stats = {}

        if include_joint:
            # Runtime sum of LoRA deltas is mathematically equivalent to
            # cat-style LoRA Task Arithmetic for weights [1.0, 1.0].
            joint_mode = activate_joint_for_training(model, args)
            joint_clean_features = clean_sampler.sample(args.joint_clean_n)
            joint_poison_features = poison_sampler.sample(args.joint_poison_n)

            loss_joint_clean = forward_feature_loss(model, tokenizer, joint_clean_features, device)
            if sequential_distributed_backward:
                loss_joint_clean = _backward_sequential_objective(
                    loss_joint_clean,
                    args.lambda_joint * args.lambda_joint_clean * joint_clean_scale,
                    model,
                    args,
                    transfer_joint=True,
                )

            loss_joint_poison = forward_feature_loss(model, tokenizer, joint_poison_features, device)
            if sequential_distributed_backward:
                loss_joint_poison = _backward_sequential_objective(
                    loss_joint_poison,
                    args.lambda_joint * args.lambda_poison * joint_poison_scale,
                    model,
                    args,
                    transfer_joint=True,
                )

            if target_model_kd_batch is not None and teacher_logits is not None:
                if kd_mode == "generative":
                    (
                        loss_preservation_kd,
                        preservation_kept,
                        preservation_conf,
                        preservation_student_clean_n,
                        preservation_student_triggered_n,
                        preservation_target_stats,
                    ) = compute_generative_target_preservation_student_loss_from_teacher(
                        model,
                        tokenizer,
                        target_model_definition,
                        target_model_kd_batch,
                        teacher_logits,
                        args,
                        device,
                    )
                elif (
                    include_bind_flat
                    and getattr(args, "enable_bind_flat", False)
                    and not getattr(args, "enable_bind_sharp", False)
                ):
                    loss_preservation_kd, preservation_kept, preservation_conf = (
                        compute_target_clean_preservation_student_loss_from_teacher(
                            model,
                            tokenizer,
                            target_model_definition,
                            target_model_kd_batch,
                            target_model_labels,
                            teacher_logits,
                            args,
                            device,
                        )
                    )
                    preservation_student_clean_n = len(target_model_kd_batch)
                    preservation_student_triggered_n = 0
                    preservation_target_stats = {
                        "preservation_kd_target_included": 0.0,
                        "preservation_kd_candidate_count": float(len(target_model_labels)),
                    }
                else:
                    (
                        loss_preservation_kd,
                        preservation_kept,
                        preservation_conf,
                        preservation_student_clean_n,
                        preservation_student_triggered_n,
                        preservation_target_stats,
                    ) = compute_target_preservation_student_losses_from_teacher(
                        model,
                        tokenizer,
                        target_model_definition,
                        target_model_kd_batch,
                        target_model_labels,
                        teacher_logits,
                        args,
                        device,
                    )

                if sequential_distributed_backward and loss_preservation_kd is not None:
                    loss_preservation_kd = _backward_sequential_objective(
                        loss_preservation_kd,
                        args.lambda_dormant_model_kd * preservation_kd_scale,
                        model,
                        args,
                        transfer_joint=True,
                    )

        # Sample each objective's gradient before its later routing, clipping,
        # or composition with the other objectives.  Restrict this extra pair
        # of autograd calls to the configured logging cadence.
        joint_gradient_info = (
            _joint_loss_gradient_cosines(model, args, loss_joint_clean, loss_joint_poison)
            if include_joint and log_this_step and not sequential_distributed_backward
            else {}
        )

        # ----------------------------------------------------------
        # Optional TargetModel-conditioned flatness/sharpness regularization.
        # Bind Sharp reproduces the historical specific objective by maximizing
        # the perturbed joint-poison loss. Bind Flat minimizes that same
        # perturbed loss and uses clean-only native-label KD above.
        # The actual perturbation and second forward are delayed until after the
        # normal original-graph backward passes.  This avoids modifying leaf
        # parameters between forward() and backward() for the original losses.
        # ----------------------------------------------------------
        bind_objective = _weighted_sum(
            [(loss_joint_poison, args.lambda_joint * args.lambda_poison * joint_poison_scale)]
        )
        bind_flat_enabled = (
            include_bind_flat
            and getattr(args, "enable_bind_flat", False)
            and not getattr(args, "enable_bind_sharp", False)
        )
        bind_sharp_enabled = include_bind_flat and getattr(args, "enable_bind_sharp", False)
        do_bind_flat = (
            bind_flat_enabled
            and bind_objective is not None
            and step >= args.bind_flat_start_step
            and (step % max(args.bind_flat_every, 1) == 0)
            and args.bind_flat_beta != 0
            and args.bind_flat_rho > 0
        )
        do_bind_sharp = (
            bind_sharp_enabled
            and bind_objective is not None
            and step >= args.bind_sharp_start_step
            and (step % max(args.bind_sharp_every, 1) == 0)
            and args.bind_sharp_beta != 0
            and args.bind_sharp_rho > 0
        )
        do_bind_regularizer = do_bind_flat or do_bind_sharp
        bind_flat_named_params = []
        bind_flat_grads = []
        bind_flat_value = None
        bind_flat_grad_norm = None
        bind_sharp_value = None
        bind_sharp_grad_norm = None
        if do_bind_regularizer:
            bind_flat_named_params = get_active_lora_named_params(model)
            bind_flat_params = [param for _, param in bind_flat_named_params]
            if bind_flat_params:
                bind_flat_grads = torch.autograd.grad(
                    bind_objective,
                    bind_flat_params,
                    retain_graph=True,
                    allow_unused=True,
                )
                bind_grad_norm = loss_to_float(grad_l2_norm(bind_flat_grads))
                if do_bind_sharp:
                    bind_sharp_grad_norm = bind_grad_norm
                else:
                    bind_flat_grad_norm = bind_grad_norm

        if include_joint and not sequential_distributed_backward:
            backward_if_not_none(
                loss_joint_clean,
                args.lambda_joint * args.lambda_joint_clean * joint_clean_scale,
            )
            backward_if_not_none(
                loss_joint_poison,
                args.lambda_joint * args.lambda_poison * joint_poison_scale,
            )

        # ----------------------------------------------------------
        # Preservation KD only:
        #   teacher = base + frozen TargetModel LoRA on clean TargetModel input;
        #   student = base + frozen TargetModel LoRA + trainable DormantModel
        #   on the configured student view. Bind Sharp (and the legacy
        #   no-flat path) uses the historical mixed target-aware view;
        #   activation-preserving Bind Flat uses clean-only native labels.
        # ----------------------------------------------------------
        if (
            not sequential_distributed_backward
            and loss_preservation_kd is not None
            and args.lambda_dormant_model_kd != 0
        ):
            (loss_preservation_kd * float(args.lambda_dormant_model_kd) * preservation_kd_scale).backward()

        if do_bind_regularizer and bind_flat_named_params and bind_flat_grads:
            perturb = apply_grad_perturbation(
                bind_flat_named_params,
                bind_flat_grads,
                rho=args.bind_sharp_rho if do_bind_sharp else args.bind_flat_rho,
            )
            if perturb:
                try:
                    loss_joint_poison_pert = forward_feature_loss(
                        model,
                        tokenizer,
                        joint_poison_features,
                        device,
                    )
                    if loss_joint_poison_pert is not None:
                        beta = args.bind_sharp_beta if do_bind_sharp else args.bind_flat_beta
                        if do_bind_sharp:
                            (-float(beta) * joint_poison_scale * loss_joint_poison_pert).backward()
                            bind_sharp_value = float(loss_joint_poison_pert.detach().cpu())
                        else:
                            (float(beta) * joint_poison_scale * loss_joint_poison_pert).backward()
                            bind_flat_value = float(loss_joint_poison_pert.detach().cpu())
                finally:
                    restore_perturbation(bind_flat_named_params, perturb)

        if include_joint:
            if sequential_distributed_backward:
                activate_dormant_model_only(model, args)
            else:
                finish_joint_training_backward(model, args)
        sync_replicated_trainable_grads(
            dormant_model_params,
            getattr(args, "_distributed_state", None),
        )
        grad_norm, preservation_grad_norm, backdoor_grad_norm = _clip_training_grads(
            args,
            dormant_model_params,
        )
        optimizer.step()
        scheduler.step()
        if checkpoint_callback is not None:
            checkpoint_callback(step + 1, global_step_offset + step + 1)
        if log_this_step:
            preservation_weighted = _sum_loss_values(
                _weighted_loss_value(loss_solo_clean, args.lambda_solo * args.lambda_solo_clean * solo_clean_scale),
                _weighted_loss_value(loss_joint_clean, args.lambda_joint * args.lambda_joint_clean * joint_clean_scale),
                _weighted_loss_value(loss_preservation_kd, args.lambda_dormant_model_kd * preservation_kd_scale),
            )
            backdoor_weighted = _sum_loss_values(
                _weighted_loss_value(loss_solo_triggered_clean, args.lambda_solo * args.lambda_inv * solo_triggered_clean_scale),
                _weighted_loss_value(loss_joint_poison, args.lambda_joint * args.lambda_poison * joint_poison_scale),
            )
            bind_flat_weighted = None
            if bind_flat_value is not None:
                bind_flat_weighted = float(args.bind_flat_beta) * joint_poison_scale * float(bind_flat_value)
            bind_sharp_weighted = None
            if bind_sharp_value is not None:
                bind_sharp_weighted = -float(args.bind_sharp_beta) * joint_poison_scale * float(bind_sharp_value)
            append_training_loss_record(
                args,
                {
                    "stage": stage_name,
                    "stage_step": step + 1,
                    "global_step": global_step_offset + step + 1,
                    "joint_mode": joint_mode,
                    "lr": _current_lr(optimizer),
                    "grad_norm": grad_norm,
                    "preservation_grad_norm": preservation_grad_norm,
                    "backdoor_grad_norm": backdoor_grad_norm,
                    "loss_functional_clean": loss_to_float(loss_solo_clean),
                    "loss_solo_clean": loss_to_float(loss_solo_clean),
                    "loss_solo_triggered_clean": loss_to_float(loss_solo_triggered_clean),
                    "loss_joint_clean": loss_to_float(loss_joint_clean),
                    "loss_joint_poison": loss_to_float(loss_joint_poison),
                    "loss_preservation_kd": loss_to_float(loss_preservation_kd),
                    "loss_bind_flat_perturbed": bind_flat_value,
                    "loss_bind_sharp_perturbed": bind_sharp_value,
                    "loss_preservation_weighted": preservation_weighted,
                    "loss_backdoor_weighted": backdoor_weighted,
                    "loss_bind_flat_weighted": bind_flat_weighted,
                    "loss_bind_sharp_weighted": bind_sharp_weighted,
                    "loss_total": _sum_loss_values(
                        preservation_weighted,
                        backdoor_weighted,
                        bind_flat_weighted,
                        bind_sharp_weighted,
                    ),
                    "bind_flat_grad_norm": bind_flat_grad_norm,
                    "bind_sharp_grad_norm": bind_sharp_grad_norm,
                    "preservation_kept": preservation_kept,
                    "preservation_conf": preservation_conf,
                    "preservation_student_clean_n": preservation_student_clean_n,
                    "preservation_student_triggered_n": preservation_student_triggered_n,
                    **joint_gradient_info,
                    **preservation_target_stats,
                },
            )
    return global_step_offset + stage_steps


def train_standalone_alignment(
    model,
    tokenizer,
    clean_sampler: InfiniteSampler,
    triggered_clean_sampler: InfiniteSampler,
    poison_sampler: InfiniteSampler,
    optimizer,
    scheduler,
    args,
    dormant_model_params,
    device,
    global_step_offset: int = 0,
):
    return _train_cumulative_stage(
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
        stage_name="stage1_standalone_alignment",
        stage_title="STAGE 1: Standalone DormantModel alignment",
        stage_steps=args.stage1_steps,
        global_step_offset=global_step_offset,
        include_joint=False,
        include_kd=False,
        include_bind_flat=False,
        enable_grad_balance=False,
    )


def train_joint_backdoor_binding(
    model,
    tokenizer,
    clean_sampler: InfiniteSampler,
    triggered_clean_sampler: InfiniteSampler,
    poison_sampler: InfiniteSampler,
    optimizer,
    scheduler,
    args,
    dormant_model_params,
    device,
    global_step_offset: int = 0,
):
    return _train_cumulative_stage(
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
        stage_name="stage12_joint_alignment_and_backdoor_binding",
        stage_title="STAGE 1+2: Joint alignment and TargetModel + DormantModel backdoor binding",
        stage_steps=args.stage1_steps + args.stage2_steps,
        global_step_offset=global_step_offset,
        include_joint=True,
        include_kd=False,
        include_bind_flat=False,
        # Stage 2 establishes the backdoor signal; do not attenuate it by
        # norm-balancing against the larger preservation subspace.
        enable_grad_balance=False,
    )


def train_stealth_refinement(
    model,
    tokenizer,
    clean_sampler: InfiniteSampler,
    triggered_clean_sampler: InfiniteSampler,
    poison_sampler: InfiniteSampler,
    optimizer,
    scheduler,
    args,
    dormant_model_params,
    device,
    global_step_offset: int = 0,
    target_model_definition: Optional[str] = None,
    target_model_sampler: Optional[InfiniteSampler] = None,
    target_model_labels: Optional[Sequence[str]] = None,
    checkpoint_callback: Optional[Callable[[int, int], None]] = None,
):
    return _train_cumulative_stage(
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
        stage_name="stage3_stealth_refinement",
        stage_title="STAGE 3: Stealth refinement with TargetModel KD and bind-flat",
        stage_steps=args.stage3_steps,
        global_step_offset=global_step_offset,
        include_joint=True,
        include_kd=True,
        include_bind_flat=True,
        enable_grad_balance=True,
        target_model_definition=target_model_definition,
        target_model_sampler=target_model_sampler,
        target_model_labels=target_model_labels,
        checkpoint_callback=checkpoint_callback,
    )
