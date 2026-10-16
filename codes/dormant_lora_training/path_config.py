from __future__ import annotations

import argparse
import json
import os
import shlex
from typing import Dict, Mapping, Optional

from .config import (
    CROSS_TARGET_ADAPTER_ROOT,
    DATA_ROOT,
    DORMANT_ADAPTER_OUTPUT_ROOT,
    LOG_ROOT,
    PROJECT_ROOT,
    TARGET_ADAPTER_ROOT,
    normalize_base_model_name,
)
from .labels import dataset_key


def _empty(value: Optional[str]) -> bool:
    return value is None or str(value).strip() == ""


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def resolve_config_path(value: Optional[str], *, project_root: str = PROJECT_ROOT) -> Optional[str]:
    """Resolve a path value from the JSON config.

    Relative paths are interpreted from the repository root. The config also
    supports ${PROJECT_ROOT}, $PROJECT_ROOT, environment variables, and ~.
    """
    if _empty(value):
        return None
    text = str(value).strip()
    text = text.replace("${PROJECT_ROOT}", project_root).replace("$PROJECT_ROOT", project_root)
    text = os.path.expanduser(os.path.expandvars(text))
    if not os.path.isabs(text):
        text = os.path.join(project_root, text)
    return os.path.abspath(text)


def load_path_config(path: Optional[str]) -> Dict[str, object]:
    if _empty(path):
        return {}
    resolved = resolve_config_path(path)
    if resolved is None:
        return {}
    with open(resolved, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"path_config must contain a JSON object: {resolved}")
    payload["_path_config_path"] = resolved
    return payload


def _lookup_dataset_path(mapping: object, dataset: str) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None
    return mapping.get(dataset) or mapping.get(dataset_key(dataset))


def _lookup_target_adapter_path(mapping: object, base_model_name: str, target_dataset: str) -> Optional[str]:
    if not isinstance(mapping, Mapping):
        return None

    keys = [
        f"{base_model_name}:{target_dataset}",
        f"{base_model_name}/{target_dataset}",
        f"{base_model_name}.lora_{target_dataset}",
        target_dataset,
        f"lora_{target_dataset}",
    ]
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str):
            return value

    nested = mapping.get(base_model_name)
    if isinstance(nested, Mapping):
        return nested.get(target_dataset) or nested.get(f"lora_{target_dataset}")
    return None


def _lookup_base_model_path(config: Mapping[str, object], base_model_name: str) -> Optional[str]:
    mapping = config.get("base_model_paths")
    if isinstance(mapping, Mapping):
        value = mapping.get(base_model_name)
        if isinstance(value, str):
            return value
    model_root = config.get("model_root")
    if isinstance(model_root, str) and model_root.strip():
        root = resolve_config_path(model_root)
        if root:
            return os.path.join(root, base_model_name)
    return None


def _maybe_apply_root(args, attr: str, config: Mapping[str, object], key: str, default_value: str) -> bool:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        return False
    current = getattr(args, attr)
    if _empty(current) or _norm(str(current)) == _norm(default_value):
        setattr(args, attr, resolve_config_path(value))
        return True
    return False


def apply_path_config(args) -> None:
    """Apply optional JSON path config to parsed CLI args.

    Explicit per-path CLI arguments still win because they are already non-empty.
    Root defaults are replaced by config values when a config file is supplied.
    """
    config = load_path_config(getattr(args, "path_config", None))
    args.path_config = config.get("_path_config_path", getattr(args, "path_config", None))
    args._path_config = config
    if not config:
        return

    args.base_model_name = normalize_base_model_name(args.base_model_name)
    if _empty(args.base_model_path):
        base_path = _lookup_base_model_path(config, args.base_model_name)
        if base_path:
            args.base_model_path = resolve_config_path(base_path)

    _maybe_apply_root(args, "data_root", config, "data_root", DATA_ROOT)
    if hasattr(args, "attack_data_root"):
        _maybe_apply_root(
            args,
            "attack_data_root",
            config,
            "attack_data_root",
            os.path.join(PROJECT_ROOT, "attack_data"),
        )
    _maybe_apply_root(args, "target_adapter_root", config, "target_adapter_root", TARGET_ADAPTER_ROOT)
    _maybe_apply_root(args, "cross_target_adapter_root", config, "cross_target_adapter_root", CROSS_TARGET_ADAPTER_ROOT)
    args._path_config_dormant_output_root = _maybe_apply_root(
        args,
        "dormant_adapter_output_root",
        config,
        "dormant_adapter_output_root",
        DORMANT_ADAPTER_OUTPUT_ROOT,
    )
    args._path_config_log_root = _maybe_apply_root(args, "log_dir", config, "log_root", LOG_ROOT)

    args.dormant_model_dataset = dataset_key(args.dormant_model_dataset)
    args.target_model_dataset = dataset_key(args.target_model_dataset)

    if _empty(args.dormant_model_dataset_path):
        value = _lookup_dataset_path(config.get("dataset_paths"), args.dormant_model_dataset)
        if value:
            args.dormant_model_dataset_path = resolve_config_path(value)
    if _empty(args.target_model_dataset_path):
        value = _lookup_dataset_path(config.get("dataset_paths"), args.target_model_dataset)
        if value:
            args.target_model_dataset_path = resolve_config_path(value)
    if _empty(args.target_model_adapter_path):
        value = _lookup_target_adapter_path(
            config.get("target_adapter_paths"),
            args.base_model_name,
            args.target_model_dataset,
        )
        if value:
            args.target_model_adapter_path = resolve_config_path(value)


def shell_exports_for_config(
    path_config: str,
    base_model_name: str,
    dormant_dataset: str,
    target_dataset: str,
    current: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Resolve config values for bash wrappers.

    Values already present in current are kept, so explicit environment
    variables still override the JSON config at the shell layer.
    """
    current = current or {}
    config = load_path_config(path_config)
    if not config:
        return {}

    base_model_name = normalize_base_model_name(base_model_name)
    dormant_dataset = dataset_key(dormant_dataset)
    target_dataset = dataset_key(target_dataset)
    exports: Dict[str, str] = {"PATH_CONFIG": str(config["_path_config_path"])}

    def put(name: str, value: Optional[str]) -> None:
        if _empty(value) or not _empty(current.get(name)):
            return
        resolved = resolve_config_path(value)
        if resolved:
            exports[name] = resolved

    put("BASE_MODEL_PATH", _lookup_base_model_path(config, base_model_name))
    put("DATA_ROOT", config.get("data_root") if isinstance(config.get("data_root"), str) else None)
    put("ATTACK_DATA_ROOT", config.get("attack_data_root") if isinstance(config.get("attack_data_root"), str) else None)
    put("TARGET_ADAPTER_ROOT", config.get("target_adapter_root") if isinstance(config.get("target_adapter_root"), str) else None)
    put("CROSS_TARGET_ADAPTER_ROOT", config.get("cross_target_adapter_root") if isinstance(config.get("cross_target_adapter_root"), str) else None)
    put("DORMANT_ADAPTER_OUTPUT_ROOT", config.get("dormant_adapter_output_root") if isinstance(config.get("dormant_adapter_output_root"), str) else None)
    put("LOG_ROOT", config.get("log_root") if isinstance(config.get("log_root"), str) else None)
    # Optional mechanism-specific roots override the suite's derived sibling
    # directories when a machine-local config defines them.
    put("SPECIFICITY_ADAPTER_ROOT", config.get("specificity_adapter_root") if isinstance(config.get("specificity_adapter_root"), str) else None)
    put("SPECIFICITY_LOG_ROOT", config.get("specificity_log_root") if isinstance(config.get("specificity_log_root"), str) else None)

    put("DORMANT_MODEL_DATASET_PATH", _lookup_dataset_path(config.get("dataset_paths"), dormant_dataset))
    put("TARGET_MODEL_DATASET_PATH", _lookup_dataset_path(config.get("dataset_paths"), target_dataset))
    put(
        "TARGET_MODEL_ADAPTER_PATH",
        _lookup_target_adapter_path(config.get("target_adapter_paths"), base_model_name, target_dataset),
    )
    return exports


def print_shell_exports(exports: Mapping[str, str]) -> None:
    for key, value in exports.items():
        print(f"{key}={shlex.quote(str(value))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Path-config helper for shell wrappers.")
    parser.add_argument("--shell-exports", action="store_true")
    parser.add_argument("--path-config", "--path_config", dest="path_config", required=True)
    parser.add_argument("--base-model-name", required=True)
    parser.add_argument("--dormant-model-dataset", required=True)
    parser.add_argument("--target-model-dataset", required=True)
    parser.add_argument("--current", action="append", default=[], help="Current shell value as NAME=VALUE.")
    args = parser.parse_args()

    current = {}
    for item in args.current:
        name, sep, value = item.partition("=")
        if sep:
            current[name] = value
    exports = shell_exports_for_config(
        args.path_config,
        args.base_model_name,
        args.dormant_model_dataset,
        args.target_model_dataset,
        current=current,
    )
    if args.shell_exports:
        print_shell_exports(exports)


if __name__ == "__main__":
    main()
