#!/usr/bin/env python
"""Generate Self-Instruct synthetic KD inputs for TargetModel preservation."""

from __future__ import annotations

import argparse
import os

from dormant_lora_training.config import PROJECT_ROOT
from dormant_lora_training.synthetic_kd_inputs import generate_self_instruct_inputs, get_dataset_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Self-Instruct synthetic KD inputs.")
    parser.add_argument("dataset", type=str, help="TargetModel dataset name, e.g. agnews, imdb, mnli, multirc.")
    parser.add_argument("--output_root", type=str, default=os.path.join(PROJECT_ROOT, "synthetic_kd_inputs", "self_instruct"))
    parser.add_argument("--output_stem", type=str, default="latest")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--manifest_path", type=str, default=None)
    parser.add_argument("--coverage_path", type=str, default=None)
    parser.add_argument("--raw_output_path", type=str, default=None)
    parser.add_argument("--valid_pool_path", type=str, default=None)

    parser.add_argument("--synthetic_target_size", type=int, default=2048)
    parser.add_argument("--synthetic_pool_multiplier", type=float, default=2.0)
    parser.add_argument("--synthetic_generate_per_round", type=int, default=20)
    parser.add_argument("--synthetic_max_rounds", type=int, default=0)
    parser.add_argument("--synthetic_similarity_threshold", type=float, default=0.7)
    parser.add_argument("--synthetic_seed", type=int, default=42)
    parser.add_argument("--synthetic_in_context_examples", type=int, default=5)
    parser.add_argument("--synthetic_request_sleep", type=float, default=0.0)
    parser.add_argument("--allow_partial", action="store_true")
    parser.add_argument("--no_checkpoint", action="store_true")
    parser.add_argument("--reset_checkpoint", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    parser.add_argument("--llm_api_key", type=str, default=os.environ.get("LLM_API_KEY", ""))
    parser.add_argument("--llm_model", type=str, default=os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--llm_base_url", type=str, default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--llm_temperature", type=float, default=float(os.environ.get("LLM_TEMPERATURE", "0.9")))
    parser.add_argument("--llm_max_tokens", type=int, default=int(os.environ.get("LLM_MAX_TOKENS", "4096")))
    parser.add_argument("--llm_timeout", type=int, default=int(os.environ.get("LLM_TIMEOUT", "120")))
    return parser


def derive_paths(args) -> None:
    spec = get_dataset_spec(args.dataset)
    args.dataset = spec.name
    output_dir = os.path.join(args.output_root, spec.name)
    if args.output_path is None:
        args.output_path = os.path.join(output_dir, f"{args.output_stem}.inputs.jsonl")
    if args.manifest_path is None:
        args.manifest_path = os.path.join(output_dir, f"{args.output_stem}.manifest.json")
    if args.coverage_path is None:
        args.coverage_path = os.path.join(output_dir, f"{args.output_stem}.coverage.csv")
    if args.raw_output_path is None:
        args.raw_output_path = os.path.join(output_dir, f"{args.output_stem}.raw.jsonl")
    if args.valid_pool_path is None:
        args.valid_pool_path = os.path.join(output_dir, f"{args.output_stem}.valid_pool.jsonl")
    if args.synthetic_target_size <= 0:
        raise ValueError("synthetic_target_size must be positive.")
    if args.synthetic_pool_multiplier < 1.0:
        raise ValueError("synthetic_pool_multiplier must be >= 1.0.")
    if args.synthetic_generate_per_round <= 0:
        raise ValueError("synthetic_generate_per_round must be positive.")
    if not (0.0 <= args.synthetic_similarity_threshold <= 1.0):
        raise ValueError("synthetic_similarity_threshold must be in [0, 1].")
    if args.synthetic_in_context_examples <= 0:
        raise ValueError("synthetic_in_context_examples must be positive.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    derive_paths(args)
    generate_self_instruct_inputs(args)


if __name__ == "__main__":
    main()
