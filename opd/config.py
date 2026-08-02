from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_BACKENDS = {"adaptive_opd", "sampled_rkl"}
SUPPORTED_STRATEGIES = {"full", "esr"}
SUPPORTED_TOPK_MODES = {
    "reverse_kl",
    "forward_kl",
    "fixed_mixture",
    "prune_opd",
    "adaptive_v1",
    "adaptive_v2",
    "prune_plus_forward",
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config_with_base(config_path: str, overrides: list[str]) -> dict[str, Any]:
    """Load YAML, recursively resolving one or more base_config entries."""

    def load_one(path: Path, stack: tuple[Path, ...] = ()) -> dict[str, Any]:
        path = path.resolve()
        if path in stack:
            chain = " -> ".join(str(p) for p in (*stack, path))
            raise ValueError(f"Cyclic base_config chain: {chain}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        base_ref = raw.pop("base_config", None)
        if base_ref is None:
            return raw
        candidates = [Path(base_ref), path.parent / str(base_ref)]
        base_path = next((p for p in candidates if p.exists()), None)
        if base_path is None:
            raise FileNotFoundError(
                f"base_config={base_ref!r} not found; tried {candidates}"
            )
        return _deep_update(load_one(base_path, (*stack, path)), raw)

    path = Path(config_path)
    cfg = load_one(path)
    cfg["_loaded_config"] = str(path)

    parsed_overrides: dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --set {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        parsed = yaml.safe_load(value)
        cur = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = parsed
        parsed_overrides[key] = parsed
    cfg["_cli_overrides"] = parsed_overrides
    return cfg


def infer_effective_lengths(cfg: dict[str, Any]) -> None:
    strategy = str(cfg.get("strategy", "full"))
    if strategy == "full":
        horizon = int(cfg["full_max_new_tokens"])
        source = "full_max_new_tokens"
    elif strategy == "esr":
        horizon = int(cfg["prefix_length"])
        source = "prefix_length"
    else:
        raise ValueError(f"Unsupported strategy={strategy!r}")

    effective_max_length = int(cfg["max_length"])
    if bool(cfg.get("auto_shrink_max_length", False)):
        effective_max_length = min(
            effective_max_length,
            int(cfg["max_prompt_length"]) + horizon,
        )
    cfg["effective_max_new_tokens"] = horizon
    cfg["effective_max_new_tokens_source"] = source
    cfg["effective_max_length"] = effective_max_length


def validate_experiment_config(cfg: dict[str, Any]) -> None:
    backend = str(cfg.get("loss_backend", ""))
    strategy = str(cfg.get("strategy", ""))
    mode = str(cfg.get("opd_loss_mode", ""))

    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"loss_backend={backend!r} is invalid. Supported: {sorted(SUPPORTED_BACKENDS)}. "
            "The full-vocabulary trl_gjsd backend has been removed."
        )
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(
            f"strategy={strategy!r} is invalid. Supported: {sorted(SUPPORTED_STRATEGIES)}"
        )
    if backend == "adaptive_opd" and mode not in SUPPORTED_TOPK_MODES:
        raise ValueError(
            f"opd_loss_mode={mode!r} is invalid for adaptive_opd. "
            f"Supported: {sorted(SUPPORTED_TOPK_MODES)}"
        )
    if backend == "sampled_rkl" and mode != "sampled_rkl":
        raise ValueError(
            "sampled_rkl only supports opd_loss_mode=sampled_rkl. "
            "This strict check prevents forward/prune configs from silently running sampled RKL."
        )

    if strategy == "esr" and int(cfg.get("prefix_length", 0)) <= 0:
        raise ValueError("ESR requires prefix_length > 0")
    for key in ("reverse_top_k", "forward_top_k", "overlap_top_k"):
        if backend == "adaptive_opd" and int(cfg.get(key, 0)) <= 0:
            raise ValueError(f"{key} must be > 0")

    if mode == "fixed_mixture":
        alpha = float(cfg.get("mixture_forward_alpha", 0.5))
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("mixture_forward_alpha must be in [0, 1]")
    if mode in {"prune_opd", "prune_plus_forward"}:
        base = float(cfg.get("prune_w_base", 0.5))
        if not 0.0 <= base <= 1.0:
            raise ValueError("prune_w_base must be in [0, 1]")

    obsolete = {
        "kd_top_k": "Use reverse_top_k, forward_top_k and overlap_top_k explicitly.",
        "kl_renorm_topk": "Top-K support renormalization is always enabled.",
        "curriculum_lengths": "Curriculum support was removed.",
        "curriculum_boundaries": "Curriculum support was removed.",
        "reflection_rollout_max_tokens": "Reflection support was removed.",
        "reflection_chunk_size": "Reflection support was removed.",
        "beta": "beta belonged to the removed full-vocabulary GJSD backend.",
        "seq_kd": "seq_kd belonged to the removed full-vocabulary GJSD backend.",
        "lmbda": "Use mixture_forward_alpha or adaptive_forward_lambda.",
    }
    present = {key: msg for key, msg in obsolete.items() if key in cfg}
    if present:
        details = "; ".join(f"{k}: {v}" for k, v in present.items())
        raise ValueError(f"Obsolete config keys detected: {details}")


def find_latest_checkpoint(output_dir: str | Path) -> str | None:
    path = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    if path.exists():
        for candidate in path.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", candidate.name)
            if candidate.is_dir() and match:
                candidates.append((int(match.group(1)), candidate))
    return str(max(candidates)[1]) if candidates else None
