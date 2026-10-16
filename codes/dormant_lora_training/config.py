from __future__ import annotations

import os
import re
from typing import Dict, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def project_path(*parts: str) -> str:
    return os.path.join(PROJECT_ROOT, *parts)


def env_path(env_name: str, *default_parts: str) -> str:
    return os.path.abspath(os.environ.get(env_name, project_path(*default_parts)))


MODEL_ROOT = env_path("MERGEAC_MODEL_ROOT", "models")
DEFAULT_BASE_MODEL_NAME = "Llama-3-8B"
BASE_MODEL_NAME_TO_PATH: Dict[str, str] = {
    "Llama-3-8B": os.path.join(MODEL_ROOT, "Llama-3-8B"),
    "Qwen2.5-7B": os.path.join(MODEL_ROOT, "Qwen2.5-7B"),
    "Mistral-7B": os.path.join(MODEL_ROOT, "Mistral-7B"),
}
BASE_MODEL_NAME_ALIASES: Dict[str, str] = {
    "llama-3-8b": "Llama-3-8B",
    "llama3-8b": "Llama-3-8B",
    "llama_3_8b": "Llama-3-8B",
    "llama-3-8b-instruct": "Llama-3-8B",
    "qwen2.5-7b": "Qwen2.5-7B",
    "qwen2-5-7b": "Qwen2.5-7B",
    "qwen25-7b": "Qwen2.5-7B",
    "qwen2_5_7b": "Qwen2.5-7B",
    "mistral-7b": "Mistral-7B",
    "mistral7b": "Mistral-7B",
    "mistral_7b": "Mistral-7B",
}
BASE_MODEL_PATH = BASE_MODEL_NAME_TO_PATH[DEFAULT_BASE_MODEL_NAME]
DATA_ROOT = env_path("MERGEAC_DATA_ROOT", "data", "Long_Sequence")
TARGET_ADAPTER_ROOT = env_path("MERGEAC_TARGET_ADAPTER_ROOT", "adapters", "TargetModelAdapters")
DORMANT_ADAPTER_OUTPUT_ROOT = env_path("MERGEAC_DORMANT_ADAPTER_OUTPUT_ROOT", "adapters", "DormantModelAdapters")
CROSS_TARGET_ADAPTER_ROOT = env_path("MERGEAC_CROSS_TARGET_ADAPTER_ROOT", "adapters", "TargetModelAdapters")
LOG_ROOT = env_path("MERGEAC_LOG_ROOT", "logs", "train_dormant_lora_pairs")

def normalize_base_model_name(base_model_name: str) -> str:
    """Normalize supported base-model names while accepting common aliases."""
    raw = str(base_model_name or "").strip()
    if raw in BASE_MODEL_NAME_TO_PATH:
        return raw
    key = raw.lower().replace("_", "-")
    key = re.sub(r"\s+", "", key)
    if key in BASE_MODEL_NAME_ALIASES:
        return BASE_MODEL_NAME_ALIASES[key]
    supported = ", ".join(BASE_MODEL_NAME_TO_PATH.keys())
    raise ValueError(f"Unsupported base_model_name={base_model_name!r}. Supported values: {supported}")


def resolve_base_model_path(base_model_name: str, base_model_path: Optional[str] = None) -> str:
    """Resolve model path. Explicit base_model_path has priority over name mapping."""
    if base_model_path:
        return str(base_model_path).strip()
    canonical_name = normalize_base_model_name(base_model_name)
    return BASE_MODEL_NAME_TO_PATH[canonical_name]
