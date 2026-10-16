from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .adapters import set_adapter_checked
from .poisoning import transform_poison_input
from .prompts import _build_prompt_answer_feature, tokenize_lm_batch

def forward_loss(model, tokenizer, definition, examples, max_length, device):
    if len(examples) == 0:
        return None
    batch = tokenize_lm_batch(tokenizer, definition, examples, max_length=max_length, device=device)
    outputs = model(**batch)
    return outputs.loss



def forward_feature_loss(model, tokenizer, features, device):
    """Forward loss for pre-tokenized feature batches.

    This bypasses tokenizer calls in the training loop.  Features are produced by
    build_training_feature_bank() and padded only at batch time.
    """
    if len(features) == 0:
        return None
    batch = _pad_features(features, tokenizer, device)
    outputs = model(**batch)
    return outputs.loss


def backward_if_not_none(loss: Optional[torch.Tensor], weight: float, retain_graph: bool = False) -> float:
    if loss is None:
        return 0.0
    weighted = loss * float(weight)
    weighted.backward(retain_graph=retain_graph)
    return float(loss.detach().cpu())


def get_active_lora_named_params(model) -> List[Tuple[str, torch.nn.Parameter]]:
    """Return currently trainable LoRA parameters.

    Runtime joint mode returns the trainable DormantModel LoRA parameters.
    Cat-proxy joint mode returns the trainable cat-proxy parameters; the
    accumulated gradients are transferred back to DormantModel by
    finish_joint_training_backward().
    """
    out: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if param.requires_grad and "lora_" in name.lower():
            out.append((name, param))
    return out


def grad_l2_norm(grads: Sequence[Optional[torch.Tensor]], eps: float = 1e-12) -> torch.Tensor:
    valid = [g.detach().float().norm(p=2) for g in grads if g is not None]
    if not valid:
        return torch.tensor(0.0)
    return torch.norm(torch.stack(valid), p=2).clamp_min(eps)


def apply_grad_perturbation(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    grads: Sequence[Optional[torch.Tensor]],
    rho: float,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Perturb active LoRA params along a gradient direction.

    The perturbation is e_w = rho * grad / ||grad||.  It is intentionally
    applied only to parameters that are currently trainable, which keeps the
    frozen base model and frozen TargetModel LoRA untouched.
    """
    valid = [(name, param, grad) for (name, param), grad in zip(named_params, grads) if grad is not None]
    if not valid:
        return {}

    norm = grad_l2_norm([grad for _, _, grad in valid], eps=eps)
    scale = float(rho) / norm
    perturb: Dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for name, param, grad in valid:
            e_w = grad.detach() * scale
            param.add_(e_w.to(dtype=param.dtype))
            perturb[name] = e_w.detach()

    return perturb


def restore_perturbation(
    named_params: Sequence[Tuple[str, torch.nn.Parameter]],
    perturb: Dict[str, torch.Tensor],
) -> None:
    """Undo a perturbation created by apply_grad_perturbation()."""
    if not perturb:
        return

    param_map = dict(named_params)
    with torch.no_grad():
        for name, e_w in perturb.items():
            if name in param_map:
                param_map[name].sub_(e_w.to(dtype=param_map[name].dtype))


def _first_label_token_ids(tokenizer, labels: Sequence[str], device) -> torch.Tensor:
    """Return first answer token ids for diagnostics only.

    The training/KD path no longer relies on these ids because Mistral can map
    several leading-space labels to the same first token.
    """
    token_ids = []
    seen = {}
    for label in labels:
        ids = tokenizer(str(label).strip(), add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Could not tokenize label: {label!r}")
        tid = int(ids[0])
        if tid in seen and seen[tid] != label:
            print(f"[WARN] Labels {seen[tid]!r} and {label!r} share the same first token id {tid}.")
        seen[tid] = label
        token_ids.append(tid)
    return torch.tensor(token_ids, dtype=torch.long, device=device)


def label_index_from_output(output_text: str, labels: Sequence[str]) -> Optional[int]:
    output_text = str(output_text).strip()
    for i, label in enumerate(labels):
        if output_text == str(label).strip():
            return i
    # Allow minor formatting differences, e.g., model/data emits a longer string starting with label.
    for i, label in enumerate(labels):
        label = str(label).strip()
        if label and output_text.startswith(label):
            return i
    return None


def _pad_features(features: Sequence[Dict[str, List[int]]], tokenizer, device) -> Dict[str, torch.Tensor]:
    max_len = max(len(f["input_ids"]) for f in features)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids, attention_mask, labels = [], [], []
    for f in features:
        ids = list(f["input_ids"])
        lbs = list(f["labels"])
        pad_len = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        labels.append(lbs + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def _sequence_label_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-example NLL over non--100 answer tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    token_loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size).float(),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.shape)
    mask = shift_labels.ne(-100)
    token_counts = mask.sum(dim=1).clamp_min(1)
    nll_sum = (token_loss * mask.float()).sum(dim=1)
    return nll_sum, token_counts.float()


def compute_sequence_label_scores(
    model,
    tokenizer,
    definition: str,
    instances: Sequence[Dict[str, str]],
    labels: Sequence[str],
    max_length: int,
    device,
    normalize: str = "mean",
    add_eos: bool = False,
    overflow_trigger: Optional[str] = None,
) -> torch.Tensor:
    """Compute closed-set class scores for full label strings.

    Score = negative NLL(label | prompt).  This replaces the old first-token
    shortcut and is robust when labels share a tokenizer prefix token.
    """
    if len(instances) == 0:
        raise ValueError("Empty batch for label-score computation.")
    scores = []
    for label in labels:
        features = [
            _build_prompt_answer_feature(
                tokenizer,
                definition,
                inst["input"],
                str(label),
                max_length=max_length,
                add_eos=add_eos,
                overflow_trigger=overflow_trigger,
            )
            for inst in instances
        ]
        batch = _pad_features(features, tokenizer, device)
        outputs = model(**batch)
        nll_sum, token_counts = _sequence_label_nll_from_logits(outputs.logits, batch["labels"])
        if normalize == "sum":
            nll = nll_sum
        elif normalize == "mean":
            nll = nll_sum / token_counts
        else:
            raise ValueError(f"Unsupported label score normalization: {normalize}")
        scores.append(-nll)
    return torch.stack(scores, dim=-1).float()


def build_target_aware_kd_labels(
    target_model_labels: Sequence[str],
    target_output: str,
) -> Tuple[List[str], int]:
    """Return unique TargetModel labels plus the dynamic backdoor target."""
    candidate_labels: List[str] = []
    for raw_label in list(target_model_labels) + [target_output]:
        label = str(raw_label).strip()
        if label and label not in candidate_labels:
            candidate_labels.append(label)

    target = str(target_output).strip()
    if not target:
        raise ValueError("target_output must not be empty for target-aware preservation KD.")
    if target not in candidate_labels:
        raise RuntimeError("Failed to add target_output to preservation KD candidates.")
    if len(candidate_labels) < 2:
        raise ValueError("Target-aware preservation KD requires at least two candidate labels.")
    return candidate_labels, candidate_labels.index(target)


def compute_next_token_label_logits(
    model,
    tokenizer,
    definition: str,
    instances: Sequence[Dict[str, str]],
    labels: Sequence[str],
    max_length: int,
    device,
) -> torch.Tensor:
    """Backward-compatible name for full-sequence label scores.

    The previous implementation used only the first token of each label, which is
    not valid for tokenizers where labels such as " neutral" and " entailment"
    share a leading-space token.  The returned tensor still has shape
    [batch, num_labels], so the KD code can keep using KL over class scores.
    """
    return compute_sequence_label_scores(
        model,
        tokenizer,
        definition,
        instances,
        labels,
        max_length=max_length,
        device=device,
        normalize=getattr(getattr(model, "_training_args", None), "label_score_normalize", "mean"),
        add_eos=False,
    )


@torch.no_grad()
def compute_frozen_target_teacher_logits(
    model,
    tokenizer,
    target_model_definition: str,
    target_model_batch: Sequence[Dict[str, str]],
    target_model_labels: Sequence[str],
    args,
    device,
) -> torch.Tensor:
    """Teacher scores over TargetModel labels plus the dynamic target output."""
    was_training = model.training
    model.eval()
    set_adapter_checked(model, args.target_model_adapter_name)
    candidate_labels, _ = build_target_aware_kd_labels(target_model_labels, args.target_output)
    teacher_logits = compute_sequence_label_scores(
        model,
        tokenizer,
        target_model_definition,
        target_model_batch,
        labels=candidate_labels,
        max_length=args.dormant_model_max_length,
        device=device,
        normalize=args.label_score_normalize,
        add_eos=False,
    ).detach()
    if was_training:
        model.train()
    return teacher_logits


@torch.no_grad()
def compute_frozen_target_clean_teacher_logits(
    model,
    tokenizer,
    target_model_definition: str,
    target_model_batch: Sequence[Dict[str, str]],
    target_model_labels: Sequence[str],
    args,
    device,
) -> torch.Tensor:
    """Teacher scores for clean-only TargetModel preservation.

    This intentionally scores only the original TargetModel labels.  The
    target output is excluded so the activation-preserving Bind Flat regime
    keeps clean TargetModel behavior without imposing a target-leakage
    constraint.
    """
    was_training = model.training
    model.eval()
    set_adapter_checked(model, args.target_model_adapter_name)
    teacher_logits = compute_sequence_label_scores(
        model,
        tokenizer,
        target_model_definition,
        target_model_batch,
        labels=list(target_model_labels),
        max_length=args.dormant_model_max_length,
        device=device,
        normalize=args.label_score_normalize,
        add_eos=False,
    ).detach()
    if was_training:
        model.train()
    return teacher_logits


def _answer_position_logits(model, tokenizer, features, device) -> Tuple[torch.Tensor, int]:
    batch = _pad_features(features, tokenizer, device)
    outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = batch["labels"][..., 1:].contiguous()
    mask = shift_labels.ne(-100)
    if not bool(mask.any()):
        raise ValueError("Generative KD batch has no answer tokens after tokenization.")
    return shift_logits[mask], int(mask.sum().detach().cpu())


def _build_generative_kd_features(
    tokenizer,
    definition: str,
    instances: Sequence[Dict[str, str]],
    max_length: int,
    *,
    overflow_trigger: Optional[str] = None,
    allow_answer_truncation: bool = False,
):
    return [
        _build_prompt_answer_feature(
            tokenizer,
            definition,
            inst["input"],
            inst["output"],
            max_length=max_length,
            add_eos=True,
            overflow_trigger=overflow_trigger,
            allow_answer_truncation=allow_answer_truncation,
        )
        for inst in instances
    ]


@torch.no_grad()
def compute_frozen_target_teacher_token_logits(
    model,
    tokenizer,
    target_model_definition: str,
    target_model_batch: Sequence[Dict[str, str]],
    args,
    device,
) -> Dict[str, object]:
    """Teacher token distributions for generative preservation KD.

    The teacher is the frozen TargetModel adapter on clean TargetModel inputs.
    Logits are gathered only at answer-token positions from the reference
    output, which supports long open-ended answers without treating every answer
    string as a closed-set class label.
    """
    was_training = model.training
    model.eval()
    set_adapter_checked(model, args.target_model_adapter_name)
    features = _build_generative_kd_features(
        tokenizer,
        target_model_definition,
        target_model_batch,
        max_length=args.dormant_model_max_length,
        allow_answer_truncation=getattr(args, "allow_answer_truncation", False),
    )
    token_logits, token_count = _answer_position_logits(model, tokenizer, features, device)
    token_logits = token_logits.detach().float()
    temperature = float(getattr(args, "dormant_model_kd_temperature", 2.0))
    top_k = int(getattr(args, "generative_kd_top_k", 128))
    vocab_size = int(token_logits.shape[-1])
    if top_k > 0 and top_k < vocab_size:
        top_values, top_indices = torch.topk(token_logits, k=top_k, dim=-1)
        top_probs = F.softmax(top_values / temperature, dim=-1).detach()
        payload = {
            "top_indices": top_indices.detach().cpu(),
            "top_probs": top_probs.detach().cpu(),
            "token_count": token_count,
            "example_count": len(target_model_batch),
            "effective_candidates": top_k,
            "teacher_mean_conf": float(top_probs.max(dim=-1).values.mean().detach().cpu()),
        }
    else:
        probs = F.softmax(token_logits / temperature, dim=-1).detach()
        payload = {
            "full_probs": probs.cpu(),
            "token_count": token_count,
            "example_count": len(target_model_batch),
            "effective_candidates": vocab_size,
            "teacher_mean_conf": float(probs.max(dim=-1).values.mean().detach().cpu()),
        }
    del token_logits
    if was_training:
        model.train()
    return payload


def compute_generative_target_preservation_student_loss_from_teacher(
    model,
    tokenizer,
    target_model_definition: str,
    target_model_batch: Sequence[Dict[str, str]],
    teacher_token_payload: Dict[str, object],
    args,
    device,
) -> Tuple[Optional[torch.Tensor], int, float, int, int, Dict[str, Optional[float]]]:
    """Token-level generative KD for open-ended TargetModel preservation."""
    clean_n = int(getattr(args, "dormant_model_kd_clean_n", len(target_model_batch) - (len(target_model_batch) // 2)))
    triggered_n = int(getattr(args, "dormant_model_kd_triggered_n", len(target_model_batch) // 2))
    if clean_n < 0 or triggered_n < 0:
        raise ValueError("KD clean/triggered student counts must be non-negative.")
    if clean_n + triggered_n != len(target_model_batch):
        raise ValueError(
            "KD clean/triggered student counts must match the sampled KD batch; "
            f"got clean={clean_n}, triggered={triggered_n}, batch={len(target_model_batch)}."
        )

    student_batch = []
    for index, inst in enumerate(target_model_batch):
        if index < clean_n:
            student_batch.append(inst)
            continue
        student_batch.append(
            {
                **inst,
                "input": transform_poison_input(
                    inst["input"],
                    method=args.poison_method,
                    trigger=args.trigger,
                    trigger_position=args.trigger_position,
                    rng=random,
                    precomputed_input=inst.get("_poisoned_input"),
                ),
            }
        )

    overflow_trigger = args.trigger if args.poison_method in {"badchar", "badnets", "sentence"} else None
    student_features = _build_generative_kd_features(
        tokenizer,
        target_model_definition,
        student_batch,
        max_length=args.dormant_model_max_length,
        overflow_trigger=overflow_trigger,
        allow_answer_truncation=getattr(args, "allow_answer_truncation", False),
    )
    student_token_logits, student_token_count = _answer_position_logits(model, tokenizer, student_features, device)
    if int(teacher_token_payload.get("token_count", 0)) != student_token_count:
        raise ValueError(
            "Generative KD teacher/student answer-token counts differ; "
            f"teacher={teacher_token_payload.get('token_count')} student={student_token_count}."
        )

    if args.lambda_dormant_model_kd <= 0:
        return None, 0, 0.0, clean_n, triggered_n, {}

    temperature = float(args.dormant_model_kd_temperature)
    top_k = int(getattr(args, "generative_kd_top_k", 128))
    student = student_token_logits.float()

    if "top_indices" in teacher_token_payload:
        top_indices = teacher_token_payload["top_indices"].to(device=device)
        teacher_probs = teacher_token_payload["top_probs"].to(device=device)
        student_log_probs = F.log_softmax(student / temperature, dim=-1).gather(-1, top_indices)
        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="batchmean",
        ) * (temperature ** 2)
        mean_conf = float(teacher_token_payload.get("teacher_mean_conf", 0.0))
        effective_candidates = int(teacher_token_payload.get("effective_candidates", top_k))
    else:
        teacher_probs = teacher_token_payload["full_probs"].to(device=device)
        kd_loss = F.kl_div(
            F.log_softmax(student / temperature, dim=-1),
            teacher_probs,
            reduction="batchmean",
        ) * (temperature ** 2)
        mean_conf = float(teacher_token_payload.get("teacher_mean_conf", 0.0))
        effective_candidates = int(teacher_token_payload.get("effective_candidates", teacher_probs.shape[-1]))

    stats: Dict[str, Optional[float]] = {
        "preservation_kd_target_included": 0.0,
        "preservation_kd_candidate_count": float(effective_candidates),
        "preservation_generative_token_count": float(student_token_count),
        "preservation_generative_top_k": float(top_k),
    }
    return kd_loss, len(target_model_batch), mean_conf, clean_n, triggered_n, stats


def compute_target_clean_preservation_student_loss_from_teacher(
    model,
    tokenizer,
    target_model_definition: str,
    target_model_batch: Sequence[Dict[str, str]],
    target_model_labels: Sequence[str],
    teacher_logits: torch.Tensor,
    args,
    device,
) -> Tuple[Optional[torch.Tensor], int, float]:
    """Clean-only KD used by the activation-preserving Bind Flat regime."""
    student_logits = compute_sequence_label_scores(
        model,
        tokenizer,
        target_model_definition,
        target_model_batch,
        labels=list(target_model_labels),
        max_length=args.dormant_model_max_length,
        device=device,
        normalize=args.label_score_normalize,
        add_eos=False,
    )
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        teacher_conf, _ = teacher_probs.max(dim=-1)
        if args.dormant_model_teacher_conf_threshold > 0:
            mask = teacher_conf >= args.dormant_model_teacher_conf_threshold
        else:
            mask = torch.ones_like(teacher_conf, dtype=torch.bool)

    kept = int(mask.sum().detach().cpu())
    mean_conf = float(teacher_conf.mean().detach().cpu())
    if kept == 0 or args.lambda_dormant_model_kd <= 0:
        return None, kept, mean_conf

    temperature = float(args.dormant_model_kd_temperature)
    kd_loss = F.kl_div(
        F.log_softmax(student_logits[mask] / temperature, dim=-1),
        F.softmax(teacher_logits[mask] / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)
    return kd_loss, kept, mean_conf


def compute_target_preservation_student_losses_from_teacher(
    model,
    tokenizer,
    target_model_definition: str,
    target_model_batch: Sequence[Dict[str, str]],
    target_model_labels: Sequence[str],
    teacher_logits: torch.Tensor,
    args,
    device,
) -> Tuple[Optional[torch.Tensor], int, float, int, int, Dict[str, Optional[float]]]:
    """Target-aware KD in the joint TargetModel+DormantModel state.

    Teacher logits are computed on clean TargetModel inputs by
    compute_frozen_target_teacher_logits(). Student logits are computed in the
    joint TargetModel+DormantModel state on a configurable student view:
    clean TargetModel inputs, triggered TargetModel inputs, or a mixture of
    both.  Both views are aligned to the clean teacher scores for the same
    examples.  Both score TargetModel labels plus target_output, so KD directly
    penalizes target leakage on TargetModel inputs.

    Assumes the active adapter is already the joint state. Only DormantModel
    receives gradients because TargetModel and the base model are frozen.
    """
    clean_n = int(getattr(args, "dormant_model_kd_clean_n", len(target_model_batch) - (len(target_model_batch) // 2)))
    triggered_n = int(getattr(args, "dormant_model_kd_triggered_n", len(target_model_batch) // 2))
    if clean_n < 0 or triggered_n < 0:
        raise ValueError("KD clean/triggered student counts must be non-negative.")
    if clean_n + triggered_n != len(target_model_batch):
        raise ValueError(
            "KD clean/triggered student counts must match the sampled KD batch; "
            f"got clean={clean_n}, triggered={triggered_n}, batch={len(target_model_batch)}."
        )
    student_batch = []
    for index, inst in enumerate(target_model_batch):
        if index < clean_n:
            student_batch.append(inst)
            continue
        student_batch.append(
            {
                **inst,
                "input": transform_poison_input(
                    inst["input"],
                    method=args.poison_method,
                    trigger=args.trigger,
                    trigger_position=args.trigger_position,
                    rng=random,
                    precomputed_input=inst.get("_poisoned_input"),
                ),
            }
        )

    candidate_labels, target_index = build_target_aware_kd_labels(
        target_model_labels,
        args.target_output,
    )
    student_logits = compute_sequence_label_scores(
        model,
        tokenizer,
        target_model_definition,
        student_batch,
        labels=candidate_labels,
        max_length=args.dormant_model_max_length,
        device=device,
        normalize=args.label_score_normalize,
        add_eos=False,
    )

    with torch.no_grad():
        normal_indices = [index for index in range(len(candidate_labels)) if index != target_index]
        teacher_probs_plain = F.softmax(teacher_logits[:, normal_indices], dim=-1)
        teacher_conf, _ = teacher_probs_plain.max(dim=-1)
        if args.dormant_model_teacher_conf_threshold > 0:
            mask = teacher_conf >= args.dormant_model_teacher_conf_threshold
        else:
            mask = torch.ones_like(teacher_conf, dtype=torch.bool)

    kept = int(mask.sum().detach().cpu())
    mean_conf = float(teacher_conf.mean().detach().cpu())
    target_stats: Dict[str, Optional[float]] = {
        "preservation_kd_target_included": 1.0,
        "preservation_kd_candidate_count": float(len(candidate_labels)),
        "preservation_teacher_target_prob": None,
        "preservation_student_target_prob": None,
        "preservation_student_target_prob_clean": None,
        "preservation_student_target_prob_triggered": None,
        "preservation_teacher_target_margin": None,
        "preservation_student_target_margin": None,
        "preservation_student_target_margin_clean": None,
        "preservation_student_target_margin_triggered": None,
    }
    if kept == 0:
        return None, kept, mean_conf, clean_n, triggered_n, target_stats

    t_logits = teacher_logits[mask]
    s_logits = student_logits[mask]

    def selected_mean(values: torch.Tensor, selection: torch.Tensor) -> Optional[float]:
        selected = values[selection]
        if selected.numel() == 0:
            return None
        return float(selected.mean().detach().cpu())

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_target_prob = teacher_probs[:, target_index]
        student_target_prob = student_probs[:, target_index]
        teacher_target_margin = (
            teacher_logits[:, target_index] - teacher_logits[:, normal_indices].max(dim=-1).values
        )
        student_target_margin = (
            student_logits[:, target_index] - student_logits[:, normal_indices].max(dim=-1).values
        )
        clean_selection = mask.clone()
        clean_selection[clean_n:] = False
        triggered_selection = mask.clone()
        triggered_selection[:clean_n] = False
        target_stats.update(
            {
                "preservation_teacher_target_prob": selected_mean(teacher_target_prob, mask),
                "preservation_student_target_prob": selected_mean(student_target_prob, mask),
                "preservation_student_target_prob_clean": selected_mean(
                    student_target_prob,
                    clean_selection,
                ),
                "preservation_student_target_prob_triggered": selected_mean(
                    student_target_prob,
                    triggered_selection,
                ),
                "preservation_teacher_target_margin": selected_mean(teacher_target_margin, mask),
                "preservation_student_target_margin": selected_mean(student_target_margin, mask),
                "preservation_student_target_margin_clean": selected_mean(
                    student_target_margin,
                    clean_selection,
                ),
                "preservation_student_target_margin_triggered": selected_mean(
                    student_target_margin,
                    triggered_selection,
                ),
            }
        )

    kd_loss = None
    if args.lambda_dormant_model_kd > 0:
        T = float(args.dormant_model_kd_temperature)
        kd_loss = F.kl_div(
            F.log_softmax(s_logits / T, dim=-1),
            F.softmax(t_logits / T, dim=-1),
            reduction="batchmean",
        ) * (T ** 2)

    return kd_loss, kept, mean_conf, clean_n, triggered_n, target_stats
