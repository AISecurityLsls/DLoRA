from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


LOSS_FIELDS = [
    "loss_total",
    "loss_functional_clean",
    "loss_solo_clean",
    "loss_solo_triggered_clean",
    "loss_joint_clean",
    "loss_joint_poison",
    "loss_preservation_kd",
    "loss_bind_flat_perturbed",
    "loss_bind_sharp_perturbed",
]

WEIGHTED_FIELDS = [
    "loss_preservation_weighted",
    "loss_backdoor_weighted",
    "loss_bind_flat_weighted",
    "loss_bind_sharp_weighted",
]

GRAD_FIELDS = [
    "preservation_grad_norm_pre_balance",
    "backdoor_grad_norm_pre_balance",
    "grad_norm",
    "preservation_grad_norm",
    "backdoor_grad_norm",
]

LR_FIELDS = ["lr"]


def _parse_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_csv(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def load_training_loss_records(path: str) -> List[Dict[str, object]]:
    if not path or not os.path.exists(path):
        return []
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".csv":
        return _load_csv(path)
    return _load_jsonl(path)


def _default_plot_path(loss_path: str) -> str:
    root, suffix = os.path.splitext(loss_path)
    if suffix.lower() in {".jsonl", ".csv"}:
        return f"{root}.png"
    return f"{loss_path}.png"


def _series(records: Sequence[Dict[str, object]], field: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for index, record in enumerate(records, start=1):
        value = _parse_float(record.get(field))
        if value is None:
            continue
        step = _parse_float(record.get("global_step"))
        xs.append(step if step is not None else float(index))
        ys.append(value)
    return xs, ys


def _stage_boundaries(records: Sequence[Dict[str, object]]) -> List[Tuple[float, str]]:
    boundaries: List[Tuple[float, str]] = []
    previous_stage = None
    for index, record in enumerate(records, start=1):
        stage = str(record.get("stage") or "")
        if not stage or stage == previous_stage:
            continue
        step = _parse_float(record.get("global_step"))
        boundaries.append((step if step is not None else float(index), stage))
        previous_stage = stage
    return boundaries


def _plot_field_group(ax, records: Sequence[Dict[str, object]], fields: Iterable[str], ylabel: str) -> bool:
    plotted = False
    for field in fields:
        xs, ys = _series(records, field)
        if not ys:
            continue
        ax.plot(xs, ys, linewidth=1.6, label=field)
        plotted = True
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(loc="best", fontsize=8)
    return plotted


def _available_fields(records: Sequence[Dict[str, object]], fields: Iterable[str]) -> List[str]:
    return [field for field in fields if _series(records, field)[1]]


def plot_training_loss_file(
    loss_path: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    separate_loss_panels: bool = False,
) -> Optional[str]:
    records = load_training_loss_records(loss_path)
    if not records:
        print(f"[WARN] No training loss records found; skip plot: {loss_path}")
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[WARN] Could not import matplotlib; skip loss plot. Install matplotlib to enable plotting. Error: {exc!r}")
        return None

    output_path = output_path or _default_plot_path(loss_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    individual_loss_fields = _available_fields(records, [*LOSS_FIELDS, *WEIGHTED_FIELDS])
    if separate_loss_panels:
        # Debug layout: one panel per loss term, so differing scales cannot
        # hide a weak or unstable objective behind loss_total.
        panel_count = len(individual_loss_fields) + 2
        fig, axes = plt.subplots(panel_count, 1, figsize=(12, max(8, 2.7 * panel_count)), sharex=True)
        axes = list(axes)
        for ax, field in zip(axes, individual_loss_fields):
            _plot_field_group(ax, records, [field], field)
        _plot_field_group(axes[-2], records, GRAD_FIELDS, "grad norm")
        _plot_field_group(axes[-1], records, LR_FIELDS, "learning rate")
    else:
        fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
        axes = list(axes)
        _plot_field_group(axes[0], records, LOSS_FIELDS, "raw loss")
        _plot_field_group(axes[1], records, WEIGHTED_FIELDS, "weighted loss")
        _plot_field_group(axes[2], records, GRAD_FIELDS, "grad norm")
        _plot_field_group(axes[3], records, LR_FIELDS, "learning rate")
    fig.suptitle(title or "Training loss curves", fontsize=14)

    boundaries = _stage_boundaries(records)
    for ax in axes:
        for step, stage in boundaries:
            ax.axvline(step, color="black", alpha=0.12, linewidth=1.0)
    if boundaries:
        ymin, ymax = axes[0].get_ylim()
        for step, stage in boundaries:
            axes[0].text(step, ymax, stage, rotation=90, va="top", ha="right", fontsize=8, alpha=0.75)
        axes[0].set_ylim(ymin, ymax)

    axes[-1].set_xlabel("global step")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[SAVE] Training loss plot written to: {output_path}")
    return output_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot DormantModel training loss curves from JSONL or CSV logs.")
    parser.add_argument("--input", "-i", required=True, help="Path to .train_loss.jsonl or .train_loss.csv.")
    parser.add_argument("--output", "-o", default=None, help="Output PNG path. Defaults to the input stem with .png.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    parser.add_argument("--separate-loss-panels", action="store_true", help="Plot every available loss term in its own subplot.")
    args = parser.parse_args(argv)
    result = plot_training_loss_file(args.input, args.output, args.title, args.separate_loss_panels)
    if result is None:
        raise SystemExit(1)
