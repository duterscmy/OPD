#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

from opd.adaptive_kl_losses import compute_topk_opd_loss
from opd.answers import judge_correctness
from opd.behavior_markers import RolloutBehaviorAnalyzer
from opd.behavior_probabilities import summarize_next_token_probabilities
from opd.checkpoint_behavior_stats import checkpoint_step
from opd.checkpoint_logit_stats import (
    DEFAULT_CATEGORY_PRIORITY,
    aggregate_category_rows,
    aggregate_checkpoint_rows,
    aggregate_marker_signal_rows,
    aggregate_probability_rows,
    attach_correctness_transitions,
    full_distribution_metrics,
    make_sample_diagnostic,
    sparse_topk_logit_gradient_norm,
)
from opd.rollout_safety import resolve_rollout_eos


SCHEMA_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc teacher/student logit analysis for rollouts produced by "
            "eval_checkpoint_rollout_behavior.py. No new rollout is generated."
        )
    )
    parser.add_argument("experiment_dir", help="Directory containing checkpoint-N")
    parser.add_argument(
        "--rollout-output-dir",
        default=None,
        help="Default: EXPERIMENT_DIR/checkpoint_behavior_eval",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: ROLLOUT_OUTPUT_DIR/logit_diagnostics",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--teacher-model", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=None,
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-chunk-size", type=int, default=8192)
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=None,
        help="Override reverse/forward/overlap Top-K together",
    )
    parser.add_argument(
        "--checkpoint-steps",
        default=None,
        help="Optional comma-separated steps; defaults to all completed rollouts",
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional first-N sample limit for a smoke test",
    )
    parser.add_argument(
        "--fixed-prefix-source",
        default="first",
        help="none, first, or checkpoint-N (default: first)",
    )
    parser.add_argument(
        "--fixed-prefix-max-samples",
        type=int,
        default=32,
        help="0 means all samples; default 32 to bound the extra scoring cost",
    )
    parser.add_argument(
        "--category-priority",
        default=",".join(DEFAULT_CATEGORY_PRIORITY),
        help="Priority for resolving overlapping marker spans",
    )
    parser.add_argument(
        "--correctness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add simple boxed/exact/numeric correctness groups",
    )
    parser.add_argument(
        "--require-saved-token-ids",
        action="store_true",
        help="Refuse older rollout JSONL that needs text retokenization",
    )
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object at {path}:{line_number}")
        records.append(payload)
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if str(key) not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _torch_dtype(name: str) -> Any:
    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(name)]


def _model_kwargs(settings: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "trust_remote_code": bool(settings["trust_remote_code"]),
        "local_files_only": bool(settings["local_files_only"]),
        "torch_dtype": _torch_dtype(str(settings["dtype"])),
        "low_cpu_mem_usage": True,
    }
    if settings.get("attn_implementation"):
        kwargs["attn_implementation"] = settings["attn_implementation"]
    device = str(settings["device"])
    if device == "auto":
        kwargs["device_map"] = "auto"
    elif device != "cpu":
        kwargs["device_map"] = {"": device}
    return kwargs


def _is_adapter_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or any(
        (path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _load_student(checkpoint: Path, settings: Mapping[str, Any]) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs = _model_kwargs(settings)
    if _is_adapter_checkpoint(checkpoint):
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("PEFT is required to score LoRA checkpoints") from error
        base = AutoModelForCausalLM.from_pretrained(settings["base_model"], **kwargs)
        model = PeftModel.from_pretrained(
            base,
            str(checkpoint),
            is_trainable=False,
            local_files_only=bool(settings["local_files_only"]),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs)
    if str(settings["device"]) == "cpu":
        model.to("cpu")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _load_teacher(settings: Mapping[str, Any]) -> Any:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        settings["teacher_model"], **_model_kwargs(settings)
    )
    if str(settings["device"]) == "cpu":
        model.to("cpu")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def _model_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _release_model(model: Any | None = None) -> None:
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prefix_compatible(student_tokenizer: Any, teacher_tokenizer: Any) -> bool:
    try:
        if len(teacher_tokenizer) < len(student_tokenizer):
            return False
        teacher_vocab = teacher_tokenizer.get_vocab()
        return all(
            teacher_vocab.get(token) == token_id
            for token, token_id in student_tokenizer.get_vocab().items()
        )
    except Exception:
        return False


def _load_tokenizers(settings: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    from transformers import AutoTokenizer, GenerationConfig

    kwargs = {
        "trust_remote_code": bool(settings["trust_remote_code"]),
        "local_files_only": bool(settings["local_files_only"]),
        "padding_side": "right",
    }
    student = AutoTokenizer.from_pretrained(settings["base_model"], **kwargs)
    teacher = AutoTokenizer.from_pretrained(settings["teacher_model"], **kwargs)
    if student.pad_token_id is None:
        if student.eos_token_id is None:
            raise ValueError("Student tokenizer has neither pad_token_id nor eos_token_id")
        student.pad_token = student.eos_token
    if teacher.pad_token_id is None:
        teacher.pad_token = teacher.eos_token
    if not _prefix_compatible(student, teacher):
        raise ValueError(
            "This post-hoc Top-K analysis requires identical or prefix-compatible "
            "student/teacher token IDs, matching adaptive_opd training."
        )

    def generation_eos(source: str) -> Any:
        try:
            return GenerationConfig.from_pretrained(
                source,
                trust_remote_code=bool(settings["trust_remote_code"]),
                local_files_only=bool(settings["local_files_only"]),
            ).eos_token_id
        except Exception:
            return None

    eos_info = resolve_rollout_eos(
        student,
        teacher,
        student_generation_eos=generation_eos(str(settings["base_model"])),
        teacher_generation_eos=generation_eos(str(settings["teacher_model"])),
        extra_eos_tokens=settings["extra_eos_tokens"],
        include_teacher_eos=True,
    )
    return student, teacher, eos_info


def _resolve_checkpoint_path(experiment_dir: Path, name: str) -> Path:
    if experiment_dir.name == name and experiment_dir.is_dir():
        return experiment_dir
    path = experiment_dir / name
    if not path.is_dir():
        raise FileNotFoundError(
            f"Rollouts exist for {name}, but model checkpoint is missing: {path}"
        )
    return path


def _discover_rollouts(
    experiment_dir: Path,
    rollout_output_dir: Path,
    args: argparse.Namespace,
) -> list[tuple[Path, Path, list[dict[str, Any]]]]:
    requested = None
    if args.checkpoint_steps:
        requested = {
            int(piece.strip())
            for piece in str(args.checkpoint_steps).split(",")
            if piece.strip()
        }
    found: list[tuple[int, Path, Path, list[dict[str, Any]]]] = []
    for result_dir in (rollout_output_dir / "checkpoints").glob("checkpoint-*"):
        if not result_dir.is_dir() or not re.fullmatch(r"checkpoint-\d+", result_dir.name):
            continue
        step = checkpoint_step(result_dir.name)
        if requested is not None and step not in requested:
            continue
        rollout_path = result_dir / "rollouts.jsonl"
        summary_path = result_dir / "summary.json"
        if not rollout_path.exists() or not summary_path.exists():
            continue
        summary = _read_json(summary_path)
        if summary.get("status") != "complete":
            continue
        records = _read_jsonl(rollout_path)
        expected = int(summary.get("summary", {}).get("sample_count", len(records)))
        if len(records) != expected:
            raise ValueError(f"{rollout_path} has {len(records)} rows; expected {expected}")
        if args.max_samples is not None:
            if int(args.max_samples) <= 0:
                raise ValueError("--max-samples must be positive")
            records = records[: int(args.max_samples)]
        checkpoint = _resolve_checkpoint_path(experiment_dir, result_dir.name)
        found.append((step, checkpoint, result_dir, records))
    found.sort(key=lambda item: item[0])
    if args.max_checkpoints is not None:
        found = found[: max(int(args.max_checkpoints), 0)]
    if not found:
        raise FileNotFoundError(
            f"No completed checkpoint rollouts found under {rollout_output_dir / 'checkpoints'}"
        )
    if requested is not None:
        missing = requested - {item[0] for item in found}
        if missing:
            raise FileNotFoundError(f"Requested rollout steps not found: {sorted(missing)}")
    return [(checkpoint, result_dir, records) for _, checkpoint, result_dir, records in found]


def _resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
    root_config_path = experiment_dir / "resolved_config.json"
    if not root_config_path.exists() and re.fullmatch(r"checkpoint-\d+", experiment_dir.name):
        root_config_path = experiment_dir.parent / "resolved_config.json"
    root_config = _read_json(root_config_path) if root_config_path.exists() else {}
    rollout_output_dir = (
        Path(args.rollout_output_dir).expanduser().resolve()
        if args.rollout_output_dir
        else experiment_dir / "checkpoint_behavior_eval"
    )
    eval_config_path = rollout_output_dir / "resolved_eval_config.json"
    eval_config = _read_json(eval_config_path) if eval_config_path.exists() else {}
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else rollout_output_dir / "logit_diagnostics"
    )
    base_model = args.base_model or eval_config.get("base_model") or root_config.get(
        "model_name_or_path"
    )
    teacher_model = args.teacher_model or root_config.get("teacher_model_name_or_path")
    if not base_model:
        raise ValueError("Could not infer base model; pass --base-model")
    if not teacher_model:
        raise ValueError("Could not infer teacher model; pass --teacher-model")
    extra_eos = root_config.get("rollout_extra_eos_tokens", ["<|im_end|>"])
    if isinstance(extra_eos, str):
        extra_eos = [extra_eos]
    trust_remote_code = args.trust_remote_code
    if trust_remote_code is None:
        trust_remote_code = bool(root_config.get("trust_remote_code", True))
    settings = {
        "schema_version": SCHEMA_VERSION,
        "experiment_dir": str(experiment_dir),
        "rollout_output_dir": str(rollout_output_dir),
        "output_dir": str(output_dir),
        "base_model": str(base_model),
        "teacher_model": str(teacher_model),
        "device": str(args.device),
        "dtype": str(args.dtype or root_config.get("dtype", "bfloat16")),
        "batch_size": int(args.batch_size),
        "vocab_chunk_size": int(args.vocab_chunk_size),
        "trust_remote_code": bool(trust_remote_code),
        "local_files_only": bool(args.local_files_only),
        "attn_implementation": args.attn_implementation
        or root_config.get("attn_implementation"),
        "extra_eos_tokens": list(extra_eos),
        "marker_dictionary": eval_config.get("marker_dictionary")
        or root_config.get("behavior_marker_dictionary"),
        "focus_markers": eval_config.get("focus_markers")
        or root_config.get("behavior_focus_markers"),
        "loss_config": dict(root_config),
        "fixed_prefix_source": str(args.fixed_prefix_source),
        "fixed_prefix_max_samples": int(args.fixed_prefix_max_samples),
        "category_priority": [
            piece.strip()
            for piece in str(args.category_priority).split(",")
            if piece.strip()
        ],
        "correctness": bool(args.correctness),
        "require_saved_token_ids": bool(args.require_saved_token_ids),
    }
    if settings["batch_size"] <= 0 or settings["vocab_chunk_size"] <= 0:
        raise ValueError("--batch-size and --vocab-chunk-size must be positive")
    if settings["fixed_prefix_max_samples"] < 0:
        raise ValueError("--fixed-prefix-max-samples must be >= 0")
    if str(settings["loss_config"].get("loss_backend", "adaptive_opd")) != "adaptive_opd":
        raise ValueError(
            "This diagnostic currently reproduces loss_backend=adaptive_opd only"
        )
    if args.analysis_top_k is not None:
        if int(args.analysis_top_k) <= 0:
            raise ValueError("--analysis-top-k must be positive")
        for key in ("reverse_top_k", "forward_top_k", "overlap_top_k"):
            settings["loss_config"][key] = int(args.analysis_top_k)
    settings["loss_config"].setdefault("opd_loss_mode", "reverse_kl")
    settings["loss_config"].setdefault("loss_normalization", "per_sequence")
    return settings


def _reconstruct_ids(
    record: Mapping[str, Any],
    tokenizer: Any,
    *,
    require_saved: bool,
) -> tuple[list[int], list[int], str]:
    full_prompt = [
        int(value)
        for value in tokenizer.encode(str(record["prompt_text"]), add_special_tokens=False)
    ]
    prompt_length = int(record.get("prompt_length", len(full_prompt)))
    prompt_ids = full_prompt[-prompt_length:]
    saved = record.get("completion_token_ids")
    if isinstance(saved, list):
        completion_ids = [int(value) for value in saved]
        source = "saved_exact"
    else:
        if require_saved:
            raise ValueError(
                "rollouts.jsonl lacks completion_token_ids. Regenerate with "
                "--save-token-ids, or omit --require-saved-token-ids."
            )
        completion_ids = [
            int(value)
            for value in tokenizer.encode(
                str(record.get("rollout_text", "")), add_special_tokens=False
            )
        ]
        source = "retokenized_text"
    if not completion_ids:
        raise ValueError(f"Empty completion for sample {record.get('sample_id')}")
    return prompt_ids, completion_ids, source


def _collate(
    batch: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    device: Any,
    *,
    require_saved: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[int]], list[str]]:
    rows: list[list[int]] = []
    labels: list[list[int]] = []
    completions: list[list[int]] = []
    sources: list[str] = []
    for record in batch:
        prompt_ids, completion_ids, source = _reconstruct_ids(
            record, tokenizer, require_saved=require_saved
        )
        rows.append(prompt_ids + completion_ids)
        labels.append([-100] * len(prompt_ids) + completion_ids)
        completions.append(completion_ids)
        sources.append(source)
    width = max(len(row) for row in rows)
    pad = int(tokenizer.pad_token_id)
    input_ids = torch.tensor(
        [row + [pad] * (width - len(row)) for row in rows],
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.tensor(
        [[1] * len(row) + [0] * (width - len(row)) for row in rows],
        dtype=torch.long,
        device=device,
    )
    label_tensor = torch.tensor(
        [row + [-100] * (width - len(row)) for row in labels],
        dtype=torch.long,
        device=device,
    )
    return input_ids, attention_mask, label_tensor, completions, sources


def _last_logits(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    lengths = attention_mask.sum(dim=1).long().clamp_min(1)
    batch = torch.arange(int(logits.shape[0]), device=logits.device)
    return logits[batch, lengths - 1, :]


def _combine_probability_samples(
    student: Mapping[str, Any], teacher: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {"sets": {}}
    names = sorted(set(student.get("sets", {})) | set(teacher.get("sets", {})))
    for name in names:
        metrics: dict[str, float] = {}
        for prefix, source in (("student", student), ("teacher", teacher)):
            for metric, value in source.get("sets", {}).get(name, {}).items():
                metrics[f"{prefix}_{metric}"] = float(value)
        result["sets"][name] = metrics
    for prefix, source in (("student", student), ("teacher", teacher)):
        repetition = source.get("repetition_continuation")
        if repetition:
            result[f"{prefix}_repetition_continuation"] = dict(repetition)
            repeated = result["sets"].setdefault("repetition/continuation", {})
            for metric, value in repetition.items():
                repeated[f"{prefix}_{metric}"] = float(value)
    return result


def _correctness(record: Mapping[str, Any], enabled: bool) -> dict[str, Any]:
    if not enabled:
        return {
            "is_correct": None,
            "student_answer": None,
            "ground_truth_answer": None,
            "correctness_method": "disabled",
        }
    result = judge_correctness(
        str(record.get("rollout_text", "")),
        str(record.get("reference_answer", "")),
        str(record.get("reference_solution", "")),
        mode="auto",
    )
    return {
        **result,
        "correctness_method": "simple_boxed_exact_or_numeric",
    }


def _score_records(
    *,
    records: Sequence[Mapping[str, Any]],
    view: str,
    source_checkpoint: str,
    checkpoint: Path,
    student_model: Any,
    teacher_model: Any,
    tokenizer: Any,
    analyzer: RolloutBehaviorAnalyzer,
    eos_token_ids: Sequence[int],
    token_sets: Mapping[str, Sequence[int]],
    settings: Mapping[str, Any],
    analysis_fingerprint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scored: list[dict[str, Any]] = []
    marker_tokens: list[dict[str, Any]] = []
    student_device = _model_device(student_model)
    teacher_device = _model_device(teacher_model)
    if student_device != teacher_device:
        raise ValueError(
            f"Student is on {student_device}, teacher on {teacher_device}; use one scoring device"
        )
    batch_size = int(settings["batch_size"])
    loss_config = dict(settings["loss_config"])
    normalization = str(loss_config.get("loss_normalization", "per_sequence"))
    retokenized = 0

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        input_ids, attention_mask, labels, completion_ids, token_sources = _collate(
            batch,
            tokenizer,
            student_device,
            require_saved=bool(settings["require_saved_token_ids"]),
        )
        retokenized += sum(source != "saved_exact" for source in token_sources)
        with torch.inference_mode():
            truncated_weight = float(
                loss_config.get("truncated_rollout_weight", 0.0)
            )
            sequence_weights = torch.tensor(
                [
                    truncated_weight if bool(record.get("hit_horizon", False)) else 1.0
                    for record in batch
                ],
                dtype=torch.float32,
                device=student_device,
            )
            student_logits_raw = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            teacher_logits_raw = teacher_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits
            common_vocab = min(
                int(student_logits_raw.shape[-1]), int(teacher_logits_raw.shape[-1])
            )
            if int(input_ids.max().item()) >= common_vocab:
                raise ValueError(
                    f"Input token ID {int(input_ids.max().item())} is outside common vocab {common_vocab}"
                )
            output = compute_topk_opd_loss(
                student_logits_raw,
                teacher_logits_raw,
                labels,
                loss_config,
                collect_diagnostics=True,
                sequence_weights=sequence_weights,
            )
            diagnostics = output.diagnostics
            active = diagnostics["active"].bool()
            raw_gradient, training_gradient = sparse_topk_logit_gradient_norm(
                diagnostics,
                loss_normalization=normalization,
                sequence_weights=sequence_weights,
            )
            student_next = student_logits_raw[:, :-1, :common_vocab]
            teacher_next = teacher_logits_raw[:, :-1, :common_vocab]
            targets = labels[:, 1:]
            student_full = full_distribution_metrics(
                student_next,
                log_z=diagnostics["student_log_z"],
                targets=targets,
                token_set=eos_token_ids,
                vocab_chunk_size=int(settings["vocab_chunk_size"]),
            )
            teacher_full = full_distribution_metrics(
                teacher_next,
                log_z=diagnostics["teacher_log_z"],
                targets=targets,
                token_set=eos_token_ids,
                vocab_chunk_size=int(settings["vocab_chunk_size"]),
            )
            student_terminal = full_distribution_metrics(
                _last_logits(student_logits_raw, attention_mask)[:, None, :common_vocab],
                token_set=eos_token_ids,
                vocab_chunk_size=int(settings["vocab_chunk_size"]),
            )
            teacher_terminal = full_distribution_metrics(
                _last_logits(teacher_logits_raw, attention_mask)[:, None, :common_vocab],
                token_set=eos_token_ids,
                vocab_chunk_size=int(settings["vocab_chunk_size"]),
            )
            student_probability = summarize_next_token_probabilities(
                student_next,
                active,
                token_sets,
                log_z=diagnostics["student_log_z"],
                targets=targets,
                terminal_logits=_last_logits(student_logits_raw, attention_mask)[
                    :, :common_vocab
                ],
                completion_ids=completion_ids,
                repetition_ngram_size=int(
                    settings["loss_config"].get("rollout_repetition_ngram_size", 4)
                ),
            )
            teacher_probability = summarize_next_token_probabilities(
                teacher_next,
                active,
                token_sets,
                log_z=diagnostics["teacher_log_z"],
                targets=targets,
                terminal_logits=_last_logits(teacher_logits_raw, attention_mask)[
                    :, :common_vocab
                ],
                completion_ids=completion_ids,
                repetition_ngram_size=int(
                    settings["loss_config"].get("rollout_repetition_ngram_size", 4)
                ),
            )

        for batch_index, source_record in enumerate(batch):
            mask = active[batch_index]
            actual_ids = completion_ids[batch_index]
            actual_length = int(mask.sum().item())
            if actual_length != len(actual_ids):
                raise ValueError(
                    f"Active token mismatch for {source_record.get('sample_id')}: "
                    f"{actual_length} vs {len(actual_ids)}"
                )
            rollout_text = tokenizer.decode(
                actual_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            category_labels, marker_rows = analyzer.exclusive_marker_span_labels(
                actual_ids,
                rollout_text,
                eos_token_ids=eos_token_ids,
                category_priority=settings["category_priority"],
            )
            target_ids = diagnostics["targets"][batch_index]
            student_topk_hit = diagnostics["student_topk_ids"][batch_index].eq(
                target_ids.unsqueeze(-1)
            ).any(dim=-1)
            teacher_topk_hit = diagnostics["teacher_topk_ids"][batch_index].eq(
                target_ids.unsqueeze(-1)
            ).any(dim=-1)
            signed_advantage = (
                diagnostics["teacher_target_logp"]
                - diagnostics["student_target_logp"]
            )[batch_index]
            token_metrics = {
                "configured_loss": diagnostics["token_loss"][batch_index][mask],
                "reverse_loss": diagnostics["reverse_loss"][batch_index][mask],
                "forward_loss": diagnostics["forward_loss"][batch_index][mask],
                "signed_advantage": signed_advantage[mask],
                "absolute_advantage": signed_advantage.abs()[mask],
                "student_target_logp": diagnostics["student_target_logp"][batch_index][mask],
                "teacher_target_logp": diagnostics["teacher_target_logp"][batch_index][mask],
                "student_target_probability": diagnostics["student_target_logp"][batch_index][mask].exp(),
                "teacher_target_probability": diagnostics["teacher_target_logp"][batch_index][mask].exp(),
                "student_entropy": student_full["entropy"][batch_index][mask],
                "teacher_entropy": teacher_full["entropy"][batch_index][mask],
                "overlap": diagnostics["overlap"][batch_index][mask],
                "student_topk_mass": diagnostics["student_topk_mass"][batch_index][mask],
                "teacher_topk_mass": diagnostics["teacher_topk_mass"][batch_index][mask],
                "student_topk_local_entropy": diagnostics["student_topk_local_entropy"][batch_index][mask],
                "teacher_topk_local_entropy": diagnostics["teacher_topk_local_entropy"][batch_index][mask],
                "student_eos_probability": student_full["token_set_probability"][batch_index][mask],
                "teacher_eos_probability": teacher_full["token_set_probability"][batch_index][mask],
                "student_target_rank": student_full["target_rank"][batch_index][mask],
                "teacher_target_rank": teacher_full["target_rank"][batch_index][mask],
                "student_eos_rank": student_full["token_set_best_rank"][batch_index][mask],
                "teacher_eos_rank": teacher_full["token_set_best_rank"][batch_index][mask],
                "logit_gradient_proxy": raw_gradient[batch_index][mask],
                "training_weighted_logit_gradient_proxy": training_gradient[batch_index][mask],
                "target_in_student_topk": student_topk_hit[mask].float(),
                "target_in_teacher_topk": teacher_topk_hit[mask].float(),
            }
            correctness = _correctness(source_record, bool(settings["correctness"]))
            metadata = {
                "analysis_fingerprint": analysis_fingerprint,
                "view": view,
                "scoring_checkpoint": checkpoint.name,
                "scoring_checkpoint_step": checkpoint_step(checkpoint.name),
                "source_checkpoint": source_checkpoint,
                "sample_id": source_record.get("sample_id"),
                "sample_order": source_record.get("sample_order"),
                "dataset_index": source_record.get("dataset_index"),
                "subject": source_record.get("subject"),
                "level": source_record.get("level"),
                "rollout_length": len(actual_ids),
                "stop_reason": source_record.get("stop_reason"),
                "emitted_eos": source_record.get("emitted_eos"),
                "hit_horizon": source_record.get("hit_horizon"),
                "boxed_truncated": source_record.get("boxed_truncated"),
                "token_id_source": token_sources[batch_index],
                "common_vocab_size": common_vocab,
                "fixed_prefix_cohort": str(source_record.get("sample_id"))
                in set(settings.get("fixed_prefix_sample_ids", [])),
                **correctness,
            }
            probability = _combine_probability_samples(
                student_probability["samples"][batch_index],
                teacher_probability["samples"][batch_index],
            )
            terminal_metrics = {
                "terminal_student_entropy": float(
                    student_terminal["entropy"][batch_index, 0].cpu().item()
                ),
                "terminal_teacher_entropy": float(
                    teacher_terminal["entropy"][batch_index, 0].cpu().item()
                ),
                "terminal_student_eos_probability": float(
                    student_terminal["token_set_probability"][batch_index, 0]
                    .cpu()
                    .item()
                ),
                "terminal_teacher_eos_probability": float(
                    teacher_terminal["token_set_probability"][batch_index, 0]
                    .cpu()
                    .item()
                ),
                "terminal_student_eos_rank": float(
                    student_terminal["token_set_best_rank"][batch_index, 0]
                    .cpu()
                    .item()
                ),
                "terminal_teacher_eos_rank": float(
                    teacher_terminal["token_set_best_rank"][batch_index, 0]
                    .cpu()
                    .item()
                ),
            }
            sample_record, sample_markers = make_sample_diagnostic(
                metadata=metadata,
                category_labels=category_labels,
                marker_rows=marker_rows,
                token_metrics=token_metrics,
                probability_summary=probability,
                loss_normalization=normalization,
                sequence_loss_weight=float(sequence_weights[batch_index].cpu().item()),
                terminal_metrics=terminal_metrics,
            )
            scored.append(sample_record)
            marker_tokens.extend(sample_markers)
        del (
            input_ids,
            attention_mask,
            labels,
            student_logits_raw,
            teacher_logits_raw,
            output,
            diagnostics,
            sequence_weights,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[logit diagnostics] {checkpoint.name} {view}: "
            f"{min(start + batch_size, len(records))}/{len(records)}",
            flush=True,
        )
    if retokenized:
        print(
            f"[logit diagnostics] warning: {retokenized}/{len(records)} {view} "
            "rollouts lacked saved token IDs and were retokenized from text; "
            "future generation should use --save-token-ids.",
            file=sys.stderr,
            flush=True,
        )
    return scored, marker_tokens


def _analysis_fingerprint(
    settings: Mapping[str, Any],
    checkpoint: str,
    view: str,
    source_checkpoint: str,
    records: Sequence[Mapping[str, Any]],
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "base_model": settings["base_model"],
        "teacher_model": settings["teacher_model"],
        "dtype": settings["dtype"],
        "loss_config": settings["loss_config"],
        "category_priority": settings["category_priority"],
        "marker_dictionary": settings.get("marker_dictionary"),
        "focus_markers": settings.get("focus_markers"),
        "correctness": settings.get("correctness"),
        "require_saved_token_ids": settings.get("require_saved_token_ids"),
        "fixed_prefix_sample_ids": settings.get("fixed_prefix_sample_ids", []),
        "eos": settings.get("resolved_eos_token_ids"),
        "checkpoint": checkpoint,
        "view": view,
        "source_checkpoint": source_checkpoint,
        "samples": [
            {
                "sample_id": record.get("sample_id"),
                "evaluation_fingerprint": record.get("evaluation_fingerprint"),
                "rollout_length": record.get("rollout_length"),
                "rollout_content_hash": _sha256(
                    record.get("completion_token_ids", record.get("rollout_text", ""))
                ),
            }
            for record in records
        ],
    }
    return _sha256(payload)


def _load_or_score_view(
    *,
    checkpoint: Path,
    view: str,
    source_checkpoint: str,
    source_records: Sequence[Mapping[str, Any]],
    student_model: Any,
    teacher_model: Any,
    tokenizer: Any,
    analyzer: RolloutBehaviorAnalyzer,
    eos_token_ids: Sequence[int],
    token_sets: Mapping[str, Sequence[int]],
    settings: Mapping[str, Any],
    overwrite: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = Path(str(settings["output_dir"])) / "checkpoints" / checkpoint.name
    data_path = output_dir / f"{view}.jsonl"
    marker_path = output_dir / f"{view}_marker_tokens.csv"
    status_path = output_dir / f"{view}_status.json"
    fingerprint = _analysis_fingerprint(
        settings, checkpoint.name, view, source_checkpoint, source_records
    )
    if data_path.exists() and status_path.exists() and not overwrite:
        status = _read_json(status_path)
        if status.get("status") == "complete" and status.get("analysis_fingerprint") == fingerprint:
            records = _read_jsonl(data_path)
            if len(records) == len(source_records):
                print(
                    f"[logit diagnostics] reusing {checkpoint.name} {view} ({len(records)} rows)",
                    flush=True,
                )
                markers: list[dict[str, Any]] = []
                if marker_path.exists():
                    with marker_path.open("r", encoding="utf-8", newline="") as handle:
                        markers = [dict(row) for row in csv.DictReader(handle)]
                return records, markers
        raise ValueError(
            f"Existing diagnostics differ at {output_dir}. Use --overwrite or a new --output-dir."
        )
    started = time.time()
    records, markers = _score_records(
        records=source_records,
        view=view,
        source_checkpoint=source_checkpoint,
        checkpoint=checkpoint,
        student_model=student_model,
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        analyzer=analyzer,
        eos_token_ids=eos_token_ids,
        token_sets=token_sets,
        settings=settings,
        analysis_fingerprint=fingerprint,
    )
    _write_jsonl(data_path, records)
    _write_csv(marker_path, markers)
    _write_json(
        status_path,
        {
            "status": "complete",
            "analysis_fingerprint": fingerprint,
            "sample_count": len(records),
            "elapsed_seconds": time.time() - started,
            "data_path": str(data_path),
            "marker_path": str(marker_path),
        },
    )
    return records, markers


def _select_fixed_source(
    value: str,
    rollouts: Sequence[tuple[Path, Path, list[dict[str, Any]]]],
    max_samples: int,
) -> tuple[str, list[dict[str, Any]]] | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "off", "false", "0"}:
        return None
    if normalized == "first":
        selected = rollouts[0]
    else:
        selected = next(
            (item for item in rollouts if item[0].name == value), None
        )
        if selected is None:
            raise ValueError(
                f"--fixed-prefix-source={value!r} does not match completed rollouts"
            )
    records = list(selected[2])
    if max_samples > 0:
        records = records[:max_samples]
    return selected[0].name, records


def _plot_outputs(
    output_dir: Path,
    checkpoint_rows: Sequence[Mapping[str, Any]],
    category_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(f"[logit diagnostics] matplotlib unavailable: {error}", file=sys.stderr)
        return

    all_on_policy = [
        row
        for row in checkpoint_rows
        if row["view"] == "on_policy"
        and row["subset_type"] == "all"
        and row["subset"] == "all"
    ]
    all_on_policy.sort(key=lambda row: int(row["scoring_checkpoint_step"]))
    if all_on_policy:
        steps = [int(row["scoring_checkpoint_step"]) for row in all_on_policy]
        lengths = [float(row["mean_rollout_length"]) for row in all_on_policy]
        entropy = [float(row["mean_student_entropy"]) for row in all_on_policy]
        figure, axis = plt.subplots(figsize=(7.2, 4.6))
        other = axis.twinx()
        axis.plot(steps, entropy, marker="o", color="C0", label="Student entropy")
        other.plot(steps, lengths, marker="s", color="C1", label="Rollout length")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Full-vocabulary entropy (nats)", color="C0")
        other.set_ylabel("Mean rollout tokens", color="C1")
        axis.grid(alpha=0.22)
        axis.set_title("On-policy entropy and rollout length")
        figure.tight_layout()
        figure.savefig(output_dir / "entropy_and_length.png", dpi=170)
        plt.close(figure)

    selected_categories = sorted(
        {
            str(row["category"])
            for row in category_rows
            if row["view"] == "on_policy"
            and row["subset_type"] == "all"
            and row["subset"] == "all"
            and row["category"] != "other"
        }
    )
    metrics = (
        ("marker_density_per_1k", "Marker events / 1k tokens"),
        ("mean_signed_advantage", "Mean signed log-ratio advantage"),
        ("configured_loss_training_mass_share", "Train-normalized loss mass share"),
        ("gradient_training_mass_share", "Logit-gradient proxy mass share"),
    )
    if selected_categories:
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), squeeze=False)
        for axis, (metric, label) in zip(axes.flat, metrics, strict=True):
            for category in selected_categories:
                rows = [
                    row
                    for row in category_rows
                    if row["view"] == "on_policy"
                    and row["subset_type"] == "all"
                    and row["subset"] == "all"
                    and row["category"] == category
                ]
                rows.sort(key=lambda row: int(row["scoring_checkpoint_step"]))
                if rows:
                    axis.plot(
                        [int(row["scoring_checkpoint_step"]) for row in rows],
                        [float(row.get(metric, 0.0)) for row in rows],
                        marker="o",
                        markersize=3,
                        linewidth=1.2,
                        label=category.replace("_", " "),
                    )
            axis.set_xlabel("Training step")
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
        figure.suptitle("On-policy category diagnostics", y=0.995)
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
        figure.savefig(output_dir / "category_diagnostics.png", dpi=170)
        plt.close(figure)

    comparison = [
        row
        for row in checkpoint_rows
        if row["subset_type"] == "cohort" and row["subset"] == "fixed_prefix"
    ]
    if {str(row["view"]) for row in comparison} >= {"on_policy", "fixed_prefix"}:
        figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
        for view, style in (("on_policy", "-"), ("fixed_prefix", "--")):
            rows = [row for row in comparison if row["view"] == view]
            rows.sort(key=lambda row: int(row["scoring_checkpoint_step"]))
            axes[0].plot(
                [int(row["scoring_checkpoint_step"]) for row in rows],
                [float(row["mean_student_entropy"]) for row in rows],
                style,
                marker="o",
                label=view,
            )
            axes[1].plot(
                [int(row["scoring_checkpoint_step"]) for row in rows],
                [float(row["mean_signed_advantage"]) for row in rows],
                style,
                marker="o",
                label=view,
            )
        axes[0].set_ylabel("Student entropy (nats)")
        axes[1].set_ylabel("Mean signed advantage")
        for axis in axes:
            axis.set_xlabel("Training step")
            axis.grid(alpha=0.2)
            axis.legend()
        figure.suptitle("Policy shift vs state-occupancy shift")
        figure.tight_layout()
        figure.savefig(output_dir / "fixed_prefix_comparison.png", dpi=170)
        plt.close(figure)


def main() -> None:
    args = parse_args()
    if torch is None:
        raise RuntimeError(
            "PyTorch is required. Run this script in the same conda environment "
            "used for OPD training (the shell wrapper activates 'opd')."
        )
    settings = _resolve_settings(args)
    experiment_dir = Path(str(settings["experiment_dir"]))
    rollout_output_dir = Path(str(settings["rollout_output_dir"]))
    output_dir = Path(str(settings["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts = _discover_rollouts(experiment_dir, rollout_output_dir, args)
    tokenizer, teacher_tokenizer, eos_info = _load_tokenizers(settings)
    del teacher_tokenizer
    settings["resolved_eos_token_ids"] = list(eos_info.token_ids)
    analyzer = RolloutBehaviorAnalyzer(
        tokenizer,
        extra_markers=settings.get("marker_dictionary"),
        focus_markers=settings.get("focus_markers"),
    )
    token_sets = analyzer.probability_token_sets(eos_token_ids=eos_info.token_ids)
    fixed_source = _select_fixed_source(
        str(settings["fixed_prefix_source"]),
        rollouts,
        int(settings["fixed_prefix_max_samples"]),
    )
    settings["fixed_prefix_sample_ids"] = (
        [str(record.get("sample_id")) for record in fixed_source[1]]
        if fixed_source
        else []
    )
    manifest = {
        **{key: value for key, value in settings.items() if key != "loss_config"},
        "loss_config": settings["loss_config"],
        "checkpoints": [checkpoint.name for checkpoint, _, _ in rollouts],
        "on_policy_sample_counts": {
            checkpoint.name: len(records) for checkpoint, _, records in rollouts
        },
        "fixed_prefix": (
            {"source_checkpoint": fixed_source[0], "sample_count": len(fixed_source[1])}
            if fixed_source
            else None
        ),
        "eos_token_ids": list(eos_info.token_ids),
        "eos_tokens": eos_info.token_strings,
        "category_annotation": (
            "exclusive marker spans; overlaps use category_priority; all other "
            "tokens are category=other"
        ),
        "signed_advantage_definition": "log p_teacher(emitted token) - log p_student(emitted token)",
        "gradient_definition": (
            "exact L2 gradient norm with respect to selected Top-K student logits; "
            "post-hoc proxy, not parameter-gradient attribution"
        ),
        "correctness_warning": (
            "simple boxed/exact/numeric verifier; use the official benchmark "
            "verifier for publication-grade accuracy"
        ),
    }
    _write_json(output_dir / "analysis_manifest.json", manifest)
    _write_json(
        output_dir / "behavior_marker_manifest.json",
        analyzer.manifest(eos_token_ids=eos_info.token_ids),
    )

    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Rollouts: {rollout_output_dir}")
    print(f"Base model: {settings['base_model']}")
    print(f"Teacher: {settings['teacher_model']}")
    print(f"Loss mode: {settings['loss_config']['opd_loss_mode']}")
    print(f"Loss normalization: {settings['loss_config']['loss_normalization']}")
    print(f"Checkpoints: {[item[0].name for item in rollouts]}")
    print(f"Fixed-prefix source: {fixed_source[0] if fixed_source else 'disabled'}")
    print(f"Output: {output_dir}")
    print("=" * 80, flush=True)

    teacher_model = _load_teacher(settings)
    all_records: list[dict[str, Any]] = []
    all_marker_rows: list[dict[str, Any]] = []
    for checkpoint, _, on_policy_records in rollouts:
        print(f"[logit diagnostics] loading student {checkpoint.name}", flush=True)
        student_model = _load_student(checkpoint, settings)
        records, markers = _load_or_score_view(
            checkpoint=checkpoint,
            view="on_policy",
            source_checkpoint=checkpoint.name,
            source_records=on_policy_records,
            student_model=student_model,
            teacher_model=teacher_model,
            tokenizer=tokenizer,
            analyzer=analyzer,
            eos_token_ids=eos_info.token_ids,
            token_sets=token_sets,
            settings=settings,
            overwrite=bool(args.overwrite),
        )
        all_records.extend(records)
        all_marker_rows.extend(markers)
        if fixed_source is not None:
            records, markers = _load_or_score_view(
                checkpoint=checkpoint,
                view="fixed_prefix",
                source_checkpoint=fixed_source[0],
                source_records=fixed_source[1],
                student_model=student_model,
                teacher_model=teacher_model,
                tokenizer=tokenizer,
                analyzer=analyzer,
                eos_token_ids=eos_info.token_ids,
                token_sets=token_sets,
                settings=settings,
                overwrite=bool(args.overwrite),
            )
            all_records.extend(records)
            all_marker_rows.extend(markers)
        del student_model
        _release_model()

    attach_correctness_transitions(all_records)
    checkpoint_rows = aggregate_checkpoint_rows(all_records)
    category_rows = aggregate_category_rows(all_records)
    probability_rows = aggregate_probability_rows(all_records)
    marker_signal_rows = aggregate_marker_signal_rows(all_records, all_marker_rows)
    _write_jsonl(output_dir / "sample_diagnostics.jsonl", all_records)
    _write_csv(output_dir / "checkpoint_diagnostics.csv", checkpoint_rows)
    _write_csv(output_dir / "category_diagnostics.csv", category_rows)
    _write_csv(output_dir / "probability_set_diagnostics.csv", probability_rows)
    _write_csv(output_dir / "marker_token_diagnostics.csv", all_marker_rows)
    _write_csv(output_dir / "marker_signal_diagnostics.csv", marker_signal_rows)
    if args.plots:
        _plot_outputs(output_dir, checkpoint_rows, category_rows)
    _write_json(
        output_dir / "completed.json",
        {
            "status": "complete",
            "sample_view_rows": len(all_records),
            "checkpoint_rows": len(checkpoint_rows),
            "category_rows": len(category_rows),
            "probability_rows": len(probability_rows),
            "marker_signal_rows": len(marker_signal_rows),
            "output_dir": str(output_dir),
        },
    )
    del teacher_model
    _release_model()
    print(f"[logit diagnostics] completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
