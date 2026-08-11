#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from opd.behavior_markers import RolloutBehaviorAnalyzer, STRUCTURAL_PATTERNS
from opd.checkpoint_behavior_stats import (
    checkpoint_step,
    flatten_sample_record,
    marker_summary_rows,
    overall_trend_summary,
    sample_length_outputs,
    summarize_checkpoint_records,
)
from opd.rollout_safety import (
    first_complete_boxed_line_end,
    finalize_math_completion,
    repeated_ngram_ratio,
    resolve_rollout_eos,
    truncate_completion,
)


DEFAULT_DATASET = "HuggingFaceH4/MATH-500"
DEFAULT_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
SCHEMA_VERSION = 1


class StreamingBoxedDetector:
    """Incrementally find the first complete, non-empty ``\\boxed{...}``."""

    marker = "\\boxed{"

    def __init__(self) -> None:
        self.search_tail = ""
        self.in_payload = False
        self.depth = 0
        self.payload_has_content = False

    def _reset_search(self) -> None:
        self.search_tail = ""
        self.in_payload = False
        self.depth = 0
        self.payload_has_content = False

    def feed(self, text: str) -> bool:
        for character in str(text):
            if not self.in_payload:
                self.search_tail = (self.search_tail + character)[
                    -len(self.marker) :
                ]
                if self.search_tail.endswith(self.marker):
                    self.in_payload = True
                    self.depth = 1
                    self.payload_has_content = False
                    self.search_tail = ""
                continue

            if character == "{":
                self.depth += 1
                self.payload_has_content = True
            elif character == "}":
                self.depth -= 1
                if self.depth == 0:
                    if self.payload_has_content:
                        return True
                    # Ignore an echoed empty ``\\boxed{}`` placeholder.
                    self._reset_search()
            elif not character.isspace():
                self.payload_has_content = True
        return False


def _make_boxed_stopping_criteria(tokenizer: Any, prompt_width: int) -> Any:
    """Create a per-row Transformers criterion without importing torch at CLI load."""

    import torch
    from transformers import StoppingCriteria

    class _StreamingBoxedStoppingCriteria(StoppingCriteria):
        def __init__(self) -> None:
            self.detectors: dict[int, StreamingBoxedDetector] = {}
            self.seen_widths: dict[int, int] = {}
            self.stop_lengths: dict[int, int] = {}

        def __call__(
            self,
            input_ids: Any,
            scores: Any,
            **kwargs: Any,
        ) -> Any:
            del scores, kwargs
            batch_size = int(input_ids.shape[0])
            generated_width = max(int(input_ids.shape[1]) - int(prompt_width), 0)
            done = torch.zeros(
                batch_size, dtype=torch.bool, device=input_ids.device
            )
            if generated_width <= 0:
                return done

            active_indices = [
                batch_index
                for batch_index in range(batch_size)
                if batch_index not in self.stop_lengths
            ]
            active_seen = [
                self.seen_widths.get(batch_index, 0)
                for batch_index in active_indices
            ]
            minimum_seen = min(active_seen) if active_seen else generated_width
            # One device-to-host transfer per generation step, rather than one
            # synchronization per active row.
            cpu_block = (
                input_ids[
                    active_indices,
                    int(prompt_width) + minimum_seen : int(prompt_width)
                    + generated_width,
                ]
                .detach()
                .cpu()
                .tolist()
                if active_indices
                else []
            )

            for active_offset, batch_index in enumerate(active_indices):
                if batch_index in self.stop_lengths:
                    done[batch_index] = True
                    continue
                seen = self.seen_widths.get(batch_index, 0)
                if generated_width <= seen:
                    continue
                detector = self.detectors.setdefault(
                    batch_index, StreamingBoxedDetector()
                )
                new_ids = cpu_block[active_offset][seen - minimum_seen :]
                for offset, token_id in enumerate(new_ids, start=seen + 1):
                    piece = tokenizer.decode(
                        [int(token_id)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if not detector.feed(piece):
                        continue

                    # Individual-token decoding is fast. Confirm the rare hit
                    # using exact prefix decoding before stopping generation.
                    completion_ids = (
                        input_ids[
                            batch_index,
                            int(prompt_width) : int(prompt_width) + offset,
                        ]
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    exact_text = tokenizer.decode(
                        completion_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    if first_complete_boxed_line_end(exact_text) is not None:
                        self.stop_lengths[batch_index] = int(offset)
                        done[batch_index] = True
                        break
                    # A tokenizer boundary changed the decoded surface form.
                    # Rebuild state from the exact text and keep generating.
                    detector = StreamingBoxedDetector()
                    detector.feed(exact_text)
                    self.detectors[batch_index] = detector
                self.seen_widths[batch_index] = generated_width
            for batch_index in self.stop_lengths:
                done[batch_index] = True
            return done

    return _StreamingBoxedStoppingCriteria()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the same fixed MATH-500 subset on every checkpoint and "
            "aggregate rollout length and behavior-marker trends."
        )
    )
    parser.add_argument("experiment_dir", help="Directory containing checkpoint-N")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: EXPERIMENT_DIR/checkpoint_behavior_eval",
    )
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--teacher-tokenizer", default=None)
    parser.add_argument("--min-checkpoint-step", type=int, default=0)
    parser.add_argument("--max-checkpoint-step", type=int, default=None)
    parser.add_argument(
        "--checkpoint-steps",
        default=None,
        help="Optional comma-separated exact steps, for example 25,75,150",
    )
    parser.add_argument("--max-checkpoints", type=int, default=None)

    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-config-name", default=None)
    parser.add_argument("--dataset-split", default="test")
    parser.add_argument("--dataset-cache-dir", default=None)
    parser.add_argument("--problem-field", default="problem")
    parser.add_argument("--solution-field", default="solution")
    parser.add_argument("--answer-field", default="answer")
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--sample-mode", choices=("random", "first"), default="random"
    )
    parser.add_argument(
        "--rebuild-sample-set",
        action="store_true",
        help="Replace the saved fixed sample set; requires --overwrite if results exist",
    )

    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Defaults to resolved_config.json, then the boxed-answer instruction",
    )
    parser.add_argument("--user-prompt-template", default=None)
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enabled by default to match the existing loop.eval chat evaluation",
    )
    parser.add_argument(
        "--enable-thinking", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--max-prompt-length", type=int, default=None)
    parser.add_argument("--max-total-length", type=int, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default=None,
    )
    parser.add_argument(
        "--do-sample", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--generation-seed", type=int, default=1234)
    parser.add_argument(
        "--stop-after-boxed",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to the experiment config, or enabled when absent",
    )
    parser.add_argument(
        "--append-eos-after-boxed",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--include-teacher-eos",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--extra-eos-token",
        action="append",
        default=[],
        help="May be repeated; experiment-config values are retained",
    )
    parser.add_argument("--repetition-ngram-size", type=int, default=None)
    parser.add_argument("--marker-dictionary-json", default=None)
    parser.add_argument("--focus-markers-json", default=None)

    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing checkpoint rollout files",
    )
    parser.add_argument(
        "--save-token-ids",
        action="store_true",
        help="Store completion token IDs in JSONL (off by default to reduce size)",
    )
    parser.add_argument(
        "--plots", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Rebuild CSV/JSON/PNG from completed checkpoint rollouts only; "
            "does not import torch, load a model, or use a GPU"
        ),
    )
    parser.add_argument(
        "--plot-columns",
        type=int,
        default=4,
        help="Small-multiple columns for category and marker PNGs",
    )
    parser.add_argument(
        "--plot-top-markers",
        type=int,
        default=12,
        help="Number of most-changing individual markers to plot",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


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
            handle.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
    temporary.replace(path)


def _append_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, default=str) + "\n"
            )
        handle.flush()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if line_number == len(lines):
                print(
                    f"[checkpoint behavior eval] ignoring incomplete final JSONL "
                    f"line in {path}",
                    file=sys.stderr,
                )
                break
            raise
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    temporary.replace(path)


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_root_config(experiment_dir: Path) -> dict[str, Any]:
    path = experiment_dir / "resolved_config.json"
    if not path.exists():
        return {}
    return _read_json(path)


def _resolve_settings(args: argparse.Namespace) -> dict[str, Any]:
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    root_config = _load_root_config(experiment_dir)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else experiment_dir / "checkpoint_behavior_eval"
    )

    stop_after_boxed = args.stop_after_boxed
    if stop_after_boxed is None:
        stop_after_boxed = bool(
            root_config.get("rollout_truncate_after_boxed_answer", True)
        )
    append_eos = args.append_eos_after_boxed
    if append_eos is None:
        append_eos = bool(
            root_config.get("rollout_append_eos_after_boxed_answer", True)
        )
    if append_eos and not stop_after_boxed:
        raise ValueError("--append-eos-after-boxed requires --stop-after-boxed")
    include_teacher = args.include_teacher_eos
    if include_teacher is None:
        include_teacher = bool(root_config.get("rollout_include_teacher_eos", True))

    configured_extra = root_config.get("rollout_extra_eos_tokens", ["<|im_end|>"])
    if isinstance(configured_extra, str):
        configured_extra = [configured_extra]
    extra_eos = list(dict.fromkeys([*configured_extra, *args.extra_eos_token]))

    marker_dictionary = root_config.get("behavior_marker_dictionary")
    if args.marker_dictionary_json:
        marker_dictionary = _read_json(
            Path(args.marker_dictionary_json).expanduser().resolve()
        )
    focus_markers = root_config.get("behavior_focus_markers")
    if args.focus_markers_json:
        focus_markers = _read_json(Path(args.focus_markers_json).expanduser().resolve())

    settings = {
        "schema_version": SCHEMA_VERSION,
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "base_model": args.base_model or root_config.get("model_name_or_path"),
        "teacher_tokenizer": args.teacher_tokenizer
        or root_config.get("teacher_model_name_or_path"),
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "dataset_split": args.dataset_split,
        "dataset_cache_dir": args.dataset_cache_dir,
        "problem_field": args.problem_field,
        "solution_field": args.solution_field,
        "answer_field": args.answer_field,
        "num_samples": int(args.num_samples),
        "sample_seed": int(args.sample_seed),
        "sample_mode": args.sample_mode,
        "system_prompt": (
            args.system_prompt
            if args.system_prompt is not None
            else str(root_config.get("system_prompt") or DEFAULT_SYSTEM_PROMPT)
        ),
        "user_prompt_template": (
            args.user_prompt_template
            if args.user_prompt_template is not None
            else str(root_config.get("user_prompt_template") or "{problem}")
        ),
        "use_chat_template": bool(args.use_chat_template),
        "enable_thinking": bool(args.enable_thinking),
        "max_prompt_length": int(
            args.max_prompt_length
            if args.max_prompt_length is not None
            else root_config.get("max_prompt_length", 2048)
        ),
        "max_total_length": int(
            args.max_total_length
            if args.max_total_length is not None
            else root_config.get("max_length", 4096)
        ),
        "max_new_tokens": int(args.max_new_tokens),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "dtype": args.dtype or str(root_config.get("dtype", "bfloat16")),
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "top_k": int(args.top_k),
        "generation_seed": int(args.generation_seed),
        "stop_after_boxed": bool(stop_after_boxed),
        "append_eos_after_boxed": bool(append_eos),
        "include_teacher_eos": bool(include_teacher),
        "extra_eos_tokens": extra_eos,
        "repetition_ngram_size": int(
            args.repetition_ngram_size
            if args.repetition_ngram_size is not None
            else root_config.get("rollout_repetition_ngram_size", 4)
        ),
        "marker_dictionary": marker_dictionary,
        "focus_markers": focus_markers,
        "attn_implementation": args.attn_implementation
        or root_config.get("attn_implementation"),
        "trust_remote_code": (
            bool(args.trust_remote_code)
            if args.trust_remote_code is not None
            else bool(root_config.get("trust_remote_code", True))
        ),
        "local_files_only": bool(args.local_files_only),
        "save_token_ids": bool(args.save_token_ids),
        "plot_columns": int(args.plot_columns),
        "plot_top_markers": int(args.plot_top_markers),
    }
    if settings["num_samples"] <= 0:
        raise ValueError("--num-samples must be > 0")
    if settings["batch_size"] <= 0:
        raise ValueError("--batch-size must be > 0")
    if settings["max_new_tokens"] <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if settings["max_prompt_length"] <= 0 or settings["max_total_length"] <= 0:
        raise ValueError("prompt and total length limits must be > 0")
    if settings["repetition_ngram_size"] <= 0:
        raise ValueError("--repetition-ngram-size must be > 0")
    if settings["plot_columns"] <= 0:
        raise ValueError("--plot-columns must be > 0")
    if settings["plot_top_markers"] <= 0:
        raise ValueError("--plot-top-markers must be > 0")
    if settings["do_sample"] and settings["temperature"] <= 0.0:
        raise ValueError("Sampling requires --temperature > 0")
    return settings


def _discover_checkpoints(args: argparse.Namespace, experiment_dir: Path) -> list[Path]:
    requested_steps: set[int] | None = None
    if args.checkpoint_steps:
        requested_steps = {
            int(piece.strip())
            for piece in str(args.checkpoint_steps).split(",")
            if piece.strip()
        }
    candidates: list[tuple[int, Path]] = []
    if re.fullmatch(r"checkpoint-\d+", experiment_dir.name):
        candidates.append((checkpoint_step(experiment_dir.name), experiment_dir))
    else:
        for path in experiment_dir.glob("checkpoint-*"):
            if not path.is_dir() or not re.fullmatch(r"checkpoint-\d+", path.name):
                continue
            candidates.append((checkpoint_step(path.name), path))
    filtered = [
        (step, path)
        for step, path in candidates
        if step >= int(args.min_checkpoint_step)
        and (args.max_checkpoint_step is None or step <= args.max_checkpoint_step)
        and (requested_steps is None or step in requested_steps)
    ]
    filtered.sort(key=lambda item: item[0])
    if args.max_checkpoints is not None:
        filtered = filtered[: max(int(args.max_checkpoints), 0)]
    if not filtered:
        raise FileNotFoundError(
            f"No checkpoint-N directories matched under {experiment_dir}"
        )
    if requested_steps is not None:
        found = {step for step, _ in filtered}
        missing = sorted(requested_steps - found)
        if missing:
            raise FileNotFoundError(f"Requested checkpoint steps not found: {missing}")
    return [path for _, path in filtered]


def _first_nonempty(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _sample_manifest_payload(settings: Mapping[str, Any], count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": settings["dataset_name"],
        "dataset_config_name": settings["dataset_config_name"],
        "dataset_split": settings["dataset_split"],
        "problem_field": settings["problem_field"],
        "solution_field": settings["solution_field"],
        "answer_field": settings["answer_field"],
        "sample_mode": settings["sample_mode"],
        "sample_seed": settings["sample_seed"],
        "requested_samples": settings["num_samples"],
        "actual_samples": int(count),
    }


def _load_or_create_samples(
    settings: Mapping[str, Any],
    *,
    rebuild: bool,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir = Path(str(settings["output_dir"]))
    sample_path = output_dir / "selected_samples.jsonl"
    manifest_path = output_dir / "sample_set_manifest.json"
    result_root = output_dir / "checkpoints"
    existing_results = result_root.exists() and any(result_root.iterdir())
    if rebuild and existing_results and not overwrite:
        raise ValueError(
            "Checkpoint results already exist. Use --overwrite together with "
            "--rebuild-sample-set, or choose a new --output-dir."
        )
    if sample_path.exists() and not rebuild:
        samples = _read_jsonl(sample_path)
        manifest = _read_json(manifest_path) if manifest_path.exists() else {}
        expected = int(settings["num_samples"])
        if len(samples) != expected:
            raise ValueError(
                f"Saved fixed sample set has {len(samples)} rows, but --num-samples="
                f"{expected}. Reuse {len(samples)} or pass --rebuild-sample-set "
                "and --overwrite."
            )
        print(
            f"[checkpoint behavior eval] reusing fixed sample set: {sample_path}"
        )
        return samples, manifest

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The datasets package is required. Activate the same environment used "
            "for training/evaluation."
        ) from error

    load_kwargs: dict[str, Any] = {
        "split": settings["dataset_split"],
    }
    if settings.get("dataset_cache_dir"):
        load_kwargs["cache_dir"] = settings["dataset_cache_dir"]
    if settings.get("dataset_config_name"):
        dataset = load_dataset(
            settings["dataset_name"],
            settings["dataset_config_name"],
            **load_kwargs,
        )
    else:
        dataset = load_dataset(settings["dataset_name"], **load_kwargs)

    count = min(int(settings["num_samples"]), len(dataset))
    if count < int(settings["num_samples"]):
        print(
            f"[checkpoint behavior eval] dataset has only {len(dataset)} rows; "
            f"using {count}",
            file=sys.stderr,
        )
    if settings["sample_mode"] == "first":
        indices = list(range(count))
    else:
        indices = random.Random(int(settings["sample_seed"])).sample(
            range(len(dataset)), count
        )

    samples: list[dict[str, Any]] = []
    for sample_order, dataset_index in enumerate(indices):
        row = dict(dataset[int(dataset_index)])
        problem = _first_nonempty(
            row,
            [
                str(settings["problem_field"]),
                "problem",
                "question",
                "prompt",
                "query",
                "input",
            ],
        )
        if not problem:
            raise ValueError(f"No problem text found at dataset row {dataset_index}")
        solution = _first_nonempty(
            row,
            [str(settings["solution_field"]), "solution", "response", "target"],
        )
        answer = _first_nonempty(
            row,
            [str(settings["answer_field"]), "answer", "ground_truth", "final_answer"],
        )
        samples.append(
            {
                "sample_id": f"math500-{int(dataset_index):04d}",
                "sample_order": sample_order,
                "dataset_index": int(dataset_index),
                "problem": problem,
                "reference_solution": solution,
                "reference_answer": answer,
                "subject": row.get("subject"),
                "level": row.get("level"),
            }
        )
    manifest = _sample_manifest_payload(settings, len(samples))
    _write_jsonl(sample_path, samples)
    _write_json(manifest_path, manifest)
    print(f"[checkpoint behavior eval] saved fixed sample set: {sample_path}")
    return samples, manifest


def _prompt_text(tokenizer: Any, problem: str, settings: Mapping[str, Any]) -> str:
    messages: list[dict[str, str]] = []
    if str(settings["system_prompt"]):
        messages.append({"role": "system", "content": str(settings["system_prompt"])})
    user_text = str(settings["user_prompt_template"]).format(problem=problem)
    messages.append({"role": "user", "content": user_text})
    if settings["use_chat_template"] and getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(settings["enable_thinking"]),
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if not isinstance(text, str):
            raise TypeError("tokenizer.apply_chat_template(tokenize=False) was not text")
        return text
    contents = [message["content"].strip() for message in messages if message["content"].strip()]
    return "\n\n".join(contents).rstrip() + "\n\n"


def _prepare_samples(
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    max_prompt_length = int(settings["max_prompt_length"])
    for sample in samples:
        text = _prompt_text(tokenizer, str(sample["problem"]), settings)
        full_ids = [
            int(value)
            for value in tokenizer.encode(text, add_special_tokens=False)
        ]
        prompt_ids = full_ids[-max_prompt_length:]
        prepared.append(
            {
                **dict(sample),
                "prompt_text": text,
                "prompt_ids": prompt_ids,
                "prompt_truncated_tokens": max(len(full_ids) - len(prompt_ids), 0),
            }
        )
    return prepared


def _infer_base_model(checkpoints: Sequence[Path], settings: dict[str, Any]) -> str:
    if settings.get("base_model"):
        return str(settings["base_model"])
    for checkpoint in checkpoints:
        adapter_config = checkpoint / "adapter_config.json"
        if adapter_config.exists():
            base = _read_json(adapter_config).get("base_model_name_or_path")
            if base:
                settings["base_model"] = str(base)
                return str(base)
    raise ValueError(
        "Could not infer the base model. Pass --base-model; adapter checkpoints "
        "also support inference from adapter_config.json."
    )


def _load_tokenizers_and_eos(
    settings: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    from transformers import AutoTokenizer, GenerationConfig

    tokenizer_kwargs = {
        "trust_remote_code": bool(settings["trust_remote_code"]),
        "local_files_only": bool(settings["local_files_only"]),
        "padding_side": "left",
    }
    student_tokenizer = AutoTokenizer.from_pretrained(
        settings["base_model"], **tokenizer_kwargs
    )
    if student_tokenizer.pad_token_id is None:
        if student_tokenizer.eos_token_id is None:
            raise ValueError("Student tokenizer has neither pad_token_id nor eos_token_id")
        student_tokenizer.pad_token = student_tokenizer.eos_token

    try:
        student_generation = GenerationConfig.from_pretrained(
            settings["base_model"],
            trust_remote_code=bool(settings["trust_remote_code"]),
            local_files_only=bool(settings["local_files_only"]),
        )
        student_generation_eos = student_generation.eos_token_id
    except Exception as error:
        print(
            f"[checkpoint behavior eval] student generation_config unavailable: {error}",
            file=sys.stderr,
        )
        student_generation_eos = None

    teacher_tokenizer = student_tokenizer
    teacher_generation_eos = None
    teacher_source = settings.get("teacher_tokenizer")
    if settings["include_teacher_eos"] and teacher_source:
        try:
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_source, **tokenizer_kwargs
            )
            try:
                teacher_generation = GenerationConfig.from_pretrained(
                    teacher_source,
                    trust_remote_code=bool(settings["trust_remote_code"]),
                    local_files_only=bool(settings["local_files_only"]),
                )
                teacher_generation_eos = teacher_generation.eos_token_id
            except Exception:
                teacher_generation_eos = None
        except Exception as error:
            print(
                "[checkpoint behavior eval] teacher tokenizer unavailable; "
                f"continuing with student/configured EOS only: {error}",
                file=sys.stderr,
            )
            teacher_tokenizer = student_tokenizer

    eos_info = resolve_rollout_eos(
        student_tokenizer,
        teacher_tokenizer,
        student_generation_eos=student_generation_eos,
        teacher_generation_eos=teacher_generation_eos,
        extra_eos_tokens=settings["extra_eos_tokens"],
        include_teacher_eos=bool(settings["include_teacher_eos"]),
    )
    return student_tokenizer, teacher_tokenizer, eos_info


def _torch_dtype(name: str) -> Any:
    import torch

    return {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[str(name)]


def _is_adapter_checkpoint(path: Path) -> bool:
    return (path / "adapter_config.json").exists() or any(
        (path / name).exists()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    )


def _load_checkpoint_model(
    checkpoint: Path,
    settings: Mapping[str, Any],
) -> Any:
    from transformers import AutoModelForCausalLM

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": bool(settings["trust_remote_code"]),
        "local_files_only": bool(settings["local_files_only"]),
        "torch_dtype": _torch_dtype(str(settings["dtype"])),
        "low_cpu_mem_usage": True,
    }
    if settings.get("attn_implementation"):
        model_kwargs["attn_implementation"] = settings["attn_implementation"]
    device = str(settings["device"])
    if device == "auto":
        model_kwargs["device_map"] = "auto"
    elif device != "cpu":
        model_kwargs["device_map"] = {"": device}

    if _is_adapter_checkpoint(checkpoint):
        try:
            from peft import PeftModel
        except ImportError as error:
            raise RuntimeError("PEFT is required for adapter checkpoints") from error
        base = AutoModelForCausalLM.from_pretrained(
            settings["base_model"], **model_kwargs
        )
        model = PeftModel.from_pretrained(
            base,
            str(checkpoint),
            is_trainable=False,
            local_files_only=bool(settings["local_files_only"]),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint), **model_kwargs
        )
    if device == "cpu":
        model.to("cpu")
    model.eval()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    return model


def _model_input_device(model: Any) -> Any:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _model_total_limit(model: Any, settings: Mapping[str, Any]) -> int:
    configured = int(settings["max_total_length"])
    candidates = [configured]
    for config in (getattr(model, "config", None), getattr(model, "base_model", None)):
        value = getattr(config, "max_position_embeddings", None)
        if value is not None and 0 < int(value) < 10**8:
            candidates.append(int(value))
    return min(candidates)


def _left_pad(
    rows: Sequence[Sequence[int]], pad_token_id: int, device: Any
) -> tuple[Any, Any]:
    import torch

    width = max(len(row) for row in rows)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for row in rows:
        values = [int(value) for value in row]
        padding = width - len(values)
        input_rows.append([int(pad_token_id)] * padding + values)
        mask_rows.append([0] * padding + [1] * len(values))
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.long, device=device),
    )


def _generate_batch(
    model: Any,
    tokenizer: Any,
    eos_info: Any,
    analyzer: RolloutBehaviorAnalyzer,
    batch: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    checkpoint: Path,
    evaluation_fingerprint: str,
) -> list[dict[str, Any]]:
    import torch
    from transformers import StoppingCriteriaList

    device = _model_input_device(model)
    prompt_rows = [list(sample["prompt_ids"]) for sample in batch]
    input_ids, attention_mask = _left_pad(
        prompt_rows, int(tokenizer.pad_token_id), device
    )
    prompt_width = int(input_ids.shape[1])
    total_limit = _model_total_limit(model, settings)
    effective_horizon = min(
        int(settings["max_new_tokens"]), total_limit - prompt_width
    )
    if effective_horizon <= 0:
        raise ValueError(
            f"Prompt width {prompt_width} leaves no generation room under "
            f"the total length limit {total_limit}"
        )

    stopping = None
    generation_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(effective_horizon),
        "do_sample": bool(settings["do_sample"]),
        "eos_token_id": list(eos_info.token_ids),
        "pad_token_id": int(tokenizer.pad_token_id),
        "return_dict_in_generate": True,
        "use_cache": True,
    }
    if settings["do_sample"]:
        generation_kwargs.update(
            temperature=float(settings["temperature"]),
            top_p=float(settings["top_p"]),
            top_k=int(settings["top_k"]),
        )
    if settings["stop_after_boxed"]:
        stopping = _make_boxed_stopping_criteria(tokenizer, prompt_width)
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stopping])

    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    records: list[dict[str, Any]] = []
    terminal_eos_id = eos_info.preferred_teacher_eos_id()
    for batch_index, sample in enumerate(batch):
        generated_row = outputs.sequences[batch_index, prompt_width:]
        online_boxed = bool(
            stopping is not None and batch_index in stopping.stop_lengths
        )
        if online_boxed:
            generated_row = generated_row[: stopping.stop_lengths[batch_index]]
        initial = truncate_completion(
            generated_row.detach().cpu().tolist(),
            eos_info=eos_info,
            pad_token_id=tokenizer.pad_token_id,
            horizon=effective_horizon,
        )
        raw_rollout_length = len(initial.token_ids)
        finalized = finalize_math_completion(
            initial,
            tokenizer,
            repetition_ngram_size=int(settings["repetition_ngram_size"]),
            truncate_after_boxed_answer=bool(settings["stop_after_boxed"]),
            append_eos_after_boxed_answer=bool(
                settings["append_eos_after_boxed"]
            ),
            terminal_eos_token_id=int(terminal_eos_id),
            generation_stopped_after_boxed_answer=online_boxed,
            rollout_horizon=effective_horizon,
        )
        completion_ids = [int(value) for value in finalized.token_ids]
        rollout_text = tokenizer.decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        behavior = analyzer.analyze(
            completion_ids,
            rollout_text,
            eos_token_ids=eos_info.token_ids,
            repetition_ngram_size=int(settings["repetition_ngram_size"]),
        )
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "event": "checkpoint_rollout_behavior",
            "evaluation_fingerprint": evaluation_fingerprint,
            "checkpoint": checkpoint.name,
            "checkpoint_step": checkpoint_step(checkpoint.name),
            "sample_id": sample["sample_id"],
            "sample_order": sample["sample_order"],
            "dataset_index": sample["dataset_index"],
            "subject": sample.get("subject"),
            "level": sample.get("level"),
            "problem": sample["problem"],
            "reference_solution": sample.get("reference_solution", ""),
            "reference_answer": sample.get("reference_answer", ""),
            "prompt_text": sample["prompt_text"],
            "prompt_length": len(sample["prompt_ids"]),
            "prompt_truncated_tokens": sample["prompt_truncated_tokens"],
            "rollout_text": rollout_text,
            "rollout_length": len(completion_ids),
            "raw_rollout_length": raw_rollout_length,
            "requested_horizon": int(settings["max_new_tokens"]),
            "effective_horizon": int(effective_horizon),
            "total_length_limit": int(total_limit),
            "emitted_eos": finalized.emitted_eos,
            "stop_reason": finalized.stop_reason,
            "stop_token_id": finalized.stop_token_id,
            "stop_token": (
                eos_info.token_strings.get(finalized.stop_token_id, "")
                if finalized.stop_token_id is not None
                else ""
            ),
            "hit_horizon": finalized.hit_horizon,
            "raw_hit_horizon": finalized.raw_hit_horizon,
            "boxed_truncated": finalized.boxed_truncated,
            "appended_eos": finalized.appended_eos,
            "raw_boxed_count": finalized.raw_boxed_count,
            "effective_boxed_count": rollout_text.count("\\boxed{"),
            "raw_repeated_ngram_ratio": finalized.raw_repeated_ngram_ratio,
            "effective_repeated_ngram_ratio": repeated_ngram_ratio(
                completion_ids, int(settings["repetition_ngram_size"])
            ),
            "student_behavior": behavior,
        }
        if settings["save_token_ids"]:
            record["completion_token_ids"] = completion_ids
        records.append(record)
    del outputs, input_ids, attention_mask
    return records


def _evaluation_fingerprint(
    settings: Mapping[str, Any],
    sample_manifest: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    eos_info: Any,
) -> str:
    relevant_settings = {
        key: settings[key]
        for key in (
            "experiment_dir",
            "base_model",
            "dataset_name",
            "dataset_config_name",
            "dataset_split",
            "system_prompt",
            "user_prompt_template",
            "use_chat_template",
            "enable_thinking",
            "max_prompt_length",
            "max_total_length",
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "top_k",
            "generation_seed",
            "stop_after_boxed",
            "append_eos_after_boxed",
            "include_teacher_eos",
            "extra_eos_tokens",
            "repetition_ngram_size",
            "marker_dictionary",
            "focus_markers",
        )
    }
    return _sha256_json(
        {
            "settings": relevant_settings,
            "sample_manifest": dict(sample_manifest),
            "sample_ids": [sample["sample_id"] for sample in samples],
            "eos_token_ids": list(eos_info.token_ids),
            "eos_sources": eos_info.sources,
        }
    )


def _existing_records_by_sample(
    path: Path,
    sample_ids: set[str],
    evaluation_fingerprint: str,
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in _read_jsonl(path):
        sample_id = str(record.get("sample_id", ""))
        if sample_id not in sample_ids:
            raise ValueError(
                f"Existing result {path} contains a different sample: {sample_id!r}"
            )
        old_fingerprint = record.get("evaluation_fingerprint")
        if old_fingerprint != evaluation_fingerprint:
            raise ValueError(
                f"Existing result settings differ for {path}. Use --overwrite "
                "or choose a new --output-dir."
            )
        records[sample_id] = record
    return records


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _evaluate_checkpoint(
    checkpoint: Path,
    tokenizer: Any,
    eos_info: Any,
    analyzer: RolloutBehaviorAnalyzer,
    prepared_samples: Sequence[Mapping[str, Any]],
    settings: Mapping[str, Any],
    evaluation_fingerprint: str,
    *,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch
    from transformers import set_seed

    output_dir = Path(str(settings["output_dir"])) / "checkpoints" / checkpoint.name
    rollout_path = output_dir / "rollouts.jsonl"
    summary_path = output_dir / "summary.json"
    sample_ids = {str(sample["sample_id"]) for sample in prepared_samples}
    if overwrite:
        existing: dict[str, dict[str, Any]] = {}
        if rollout_path.exists():
            rollout_path.write_text("", encoding="utf-8")
    else:
        existing = _existing_records_by_sample(
            rollout_path, sample_ids, evaluation_fingerprint
        )
        if len(existing) == len(prepared_samples) and summary_path.exists():
            saved = _read_json(summary_path)
            if saved.get("evaluation_fingerprint") == evaluation_fingerprint:
                print(
                    f"[checkpoint behavior eval] skip complete {checkpoint.name}"
                )
                ordered = [existing[str(sample["sample_id"])] for sample in prepared_samples]
                return ordered, dict(saved["summary"])

    pending = [
        sample
        for sample in prepared_samples
        if str(sample["sample_id"]) not in existing
    ]
    print(
        f"[checkpoint behavior eval] loading {checkpoint.name}; "
        f"pending={len(pending)}/{len(prepared_samples)}"
    )
    started = time.time()
    set_seed(int(settings["generation_seed"]))
    model = _load_checkpoint_model(checkpoint, settings)
    try:
        batch_size = int(settings["batch_size"])
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            new_records = _generate_batch(
                model,
                tokenizer,
                eos_info,
                analyzer,
                batch,
                settings,
                checkpoint,
                evaluation_fingerprint,
            )
            _append_jsonl(rollout_path, new_records)
            for record in new_records:
                existing[str(record["sample_id"])] = record
            completed = len(existing)
            batch_mean = statistics.fmean(
                int(record["rollout_length"]) for record in new_records
            )
            print(
                f"[checkpoint behavior eval] {checkpoint.name} "
                f"{completed}/{len(prepared_samples)} "
                f"batch_mean_length={batch_mean:.1f}",
                flush=True,
            )
    finally:
        del model
        _release_model_memory()

    ordered = [existing[str(sample["sample_id"])] for sample in prepared_samples]
    # Canonicalize after a resumed run and remove any duplicate appended lines.
    _write_jsonl(rollout_path, ordered)
    summary = summarize_checkpoint_records(
        checkpoint.name,
        checkpoint_step(checkpoint.name),
        ordered,
        repetition_ngram_size=int(settings["repetition_ngram_size"]),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "evaluation_fingerprint": evaluation_fingerprint,
        "elapsed_seconds": time.time() - started,
        "summary": summary,
    }
    _write_json(summary_path, payload)
    print(
        f"[checkpoint behavior eval] completed {checkpoint.name}: "
        f"mean={summary['rollout/mean_generated_tokens']:.1f}, "
        f"median={summary['rollout/median_generated_tokens']:.1f}, "
        f"horizon_fraction={summary['rollout/truncated_fraction']:.3f}"
    )
    return ordered, summary


def _marker_universe(analyzer: RolloutBehaviorAnalyzer) -> list[tuple[str, str]]:
    markers = [
        (category, phrase)
        for category, phrases in analyzer.categories.items()
        for phrase in phrases
    ]
    markers.append(("termination", "<EOS>"))
    for name, (category, _) in STRUCTURAL_PATTERNS.items():
        markers.append((category, f"<{name}>"))
    return sorted(set(markers))


def _marker_universe_from_manifest(
    manifest: Mapping[str, Any],
) -> list[tuple[str, str]]:
    markers: list[tuple[str, str]] = []
    categories = manifest.get("categories", {})
    if isinstance(categories, Mapping):
        for category, phrases in categories.items():
            if isinstance(phrases, str):
                phrases = [phrases]
            if isinstance(phrases, Sequence):
                markers.extend(
                    (str(category), str(phrase)) for phrase in phrases
                )
    markers.append(("termination", "<EOS>"))
    for name, (category, _) in STRUCTURAL_PATTERNS.items():
        markers.append((category, f"<{name}>"))
    return sorted(set(markers))


def _write_aggregate_outputs(
    settings: Mapping[str, Any],
    checkpoint_records: Mapping[str, Sequence[Mapping[str, Any]]],
    summaries: Sequence[Mapping[str, Any]],
    *,
    marker_universe: Iterable[tuple[str, str]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir = Path(str(settings["output_dir"]))
    ordered_summaries = sorted(
        (dict(summary) for summary in summaries),
        key=lambda item: int(item["checkpoint_step"]),
    )
    _write_jsonl(output_dir / "checkpoint_summary.jsonl", ordered_summaries)
    _write_csv(output_dir / "checkpoint_summary.csv", ordered_summaries)

    long_rows = [
        flatten_sample_record(record)
        for checkpoint in sorted(
            checkpoint_records,
            key=lambda name: checkpoint_step(name),
        )
        for record in checkpoint_records[checkpoint]
    ]
    _write_csv(output_dir / "sample_checkpoint_metrics.csv", long_rows)

    steps = {name: checkpoint_step(name) for name in checkpoint_records}
    wide_rows, trend_rows = sample_length_outputs(checkpoint_records, steps)
    _write_csv(output_dir / "sample_length_trajectories.csv", wide_rows)
    _write_csv(output_dir / "sample_length_trends.csv", trend_rows)

    marker_rows = marker_summary_rows(
        checkpoint_records,
        steps,
        marker_universe=marker_universe,
    )
    _write_csv(output_dir / "behavior_marker_summary.csv", marker_rows)

    trend_summary = overall_trend_summary(ordered_summaries, trend_rows)
    _write_json(output_dir / "length_trend_summary.json", trend_summary)
    return ordered_summaries, wide_rows, marker_rows


def _load_completed_checkpoint_outputs(
    output_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_dirs = sorted(
        (
            path
            for path in checkpoint_root.glob("checkpoint-*")
            if path.is_dir() and re.fullmatch(r"checkpoint-\d+", path.name)
        ),
        key=lambda path: checkpoint_step(path.name),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(
            f"No completed checkpoint result directories found in {checkpoint_root}"
        )

    checkpoint_records: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for checkpoint_dir in checkpoint_dirs:
        summary_path = checkpoint_dir / "summary.json"
        rollout_path = checkpoint_dir / "rollouts.jsonl"
        if not summary_path.exists() or not rollout_path.exists():
            raise FileNotFoundError(
                f"Incomplete result for {checkpoint_dir.name}: expected both "
                "summary.json and rollouts.jsonl"
            )
        payload = _read_json(summary_path)
        if payload.get("status") != "complete" or not isinstance(
            payload.get("summary"), dict
        ):
            raise ValueError(f"Checkpoint result is not complete: {summary_path}")
        records = _read_jsonl(rollout_path)
        expected = int(payload["summary"].get("sample_count", len(records)))
        if len(records) != expected:
            raise ValueError(
                f"{rollout_path} contains {len(records)} records; expected {expected}"
            )
        checkpoint_records[checkpoint_dir.name] = records
        summaries.append(dict(payload["summary"]))
    return checkpoint_records, summaries


def _run_aggregate_only(
    settings: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    output_dir = Path(str(settings["output_dir"]))
    saved_config_path = output_dir / "resolved_eval_config.json"
    aggregate_settings = (
        _read_json(saved_config_path)
        if saved_config_path.exists()
        else dict(settings)
    )
    # The current path and plotting choices are presentation settings, not part
    # of the generation fingerprint. They may safely change during replotting.
    aggregate_settings["output_dir"] = str(output_dir)
    aggregate_settings["plot_columns"] = int(args.plot_columns)
    aggregate_settings["plot_top_markers"] = int(args.plot_top_markers)

    checkpoint_records, summaries = _load_completed_checkpoint_outputs(
        output_dir
    )
    manifest_path = output_dir / "behavior_marker_manifest.json"
    manifest = _read_json(manifest_path) if manifest_path.exists() else {}
    ordered_summaries, wide_rows, marker_rows = _write_aggregate_outputs(
        aggregate_settings,
        checkpoint_records,
        summaries,
        marker_universe=_marker_universe_from_manifest(manifest),
    )
    if args.plots:
        _plot_outputs(
            aggregate_settings,
            ordered_summaries,
            wide_rows,
            marker_rows,
        )
    print(
        "[checkpoint behavior eval] aggregate-only completed for "
        f"{len(ordered_summaries)} checkpoints; no model was loaded; "
        f"results written to {output_dir}",
        flush=True,
    )


def _padded_axis_limits(
    values: Sequence[float | int],
    *,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    minimum_span: float = 0.05,
    padding_fraction: float = 0.15,
) -> tuple[float, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        return (0.0, 1.0)
    low = min(numbers)
    high = max(numbers)
    data_span = high - low
    if data_span == 0.0:
        half_span = max(minimum_span / 2.0, abs(low) * 0.05, 1e-6)
        low -= half_span
        high += half_span
    else:
        padding = max(data_span * float(padding_fraction), minimum_span * 0.05)
        low -= padding
        high += padding
    if lower_bound is not None:
        low = max(low, float(lower_bound))
    if upper_bound is not None:
        high = min(high, float(upper_bound))
    if high - low < minimum_span:
        missing = minimum_span - (high - low)
        can_expand_down = lower_bound is None or low > float(lower_bound)
        can_expand_up = upper_bound is None or high < float(upper_bound)
        if can_expand_down and can_expand_up:
            low -= missing / 2.0
            high += missing / 2.0
        elif can_expand_up:
            high += missing
        elif can_expand_down:
            low -= missing
    if lower_bound is not None:
        low = max(low, float(lower_bound))
    if upper_bound is not None:
        high = min(high, float(upper_bound))
    if high <= low:
        high = low + max(minimum_span, 1e-6)
    return low, high


def _compact_step_ticks(steps: Sequence[int], maximum_ticks: int = 5) -> list[int]:
    if len(steps) <= maximum_ticks:
        return list(steps)
    indices = [
        round(index * (len(steps) - 1) / float(maximum_ticks - 1))
        for index in range(maximum_ticks)
    ]
    return [int(steps[index]) for index in dict.fromkeys(indices)]


def _set_compact_x_axis(axis: Any, steps: Sequence[int]) -> None:
    axis.set_xticks(_compact_step_ticks(steps))
    if len(steps) > 1:
        span = float(steps[-1] - steps[0])
        padding = max(span * 0.03, 1.0)
        axis.set_xlim(float(steps[0]) - padding, float(steps[-1]) + padding)


def _plot_fraction_count_small_multiples(
    plt: Any,
    *,
    output_path: Path,
    title: str,
    steps: Sequence[int],
    series: Mapping[str, Mapping[int, tuple[float, float]]],
    labels: Sequence[str],
    columns: int,
) -> None:
    if not labels:
        return
    ncols = min(max(int(columns), 1), len(labels))
    if len(labels) > ncols and len(labels) % ncols == 1 and ncols > 2:
        # Avoid a nearly empty final row (for example, 9 categories in 4
        # columns is clearer as a balanced 3x3 grid).
        ncols -= 1
    nrows = (len(labels) + ncols - 1) // ncols
    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.15 * ncols, 2.55 * nrows + 0.35),
        squeeze=False,
    )
    fraction_color = "C0"
    count_color = "C1"
    for index, label in enumerate(labels):
        row_index, column_index = divmod(index, ncols)
        axis = axes[row_index][column_index]
        count_axis = axis.twinx()
        values = series[label]
        fractions = [float(values.get(step, (0.0, 0.0))[0]) for step in steps]
        counts = [float(values.get(step, (0.0, 0.0))[1]) for step in steps]
        axis.plot(
            steps,
            fractions,
            color=fraction_color,
            marker="o",
            markersize=3.2,
            linewidth=1.4,
        )
        count_axis.plot(
            steps,
            counts,
            color=count_color,
            marker="s",
            markersize=3.0,
            linewidth=1.25,
        )
        axis.set_ylim(
            *_padded_axis_limits(
                fractions,
                lower_bound=0.0,
                upper_bound=1.0,
                minimum_span=0.06,
            )
        )
        count_axis.set_ylim(
            *_padded_axis_limits(
                counts,
                lower_bound=0.0,
                minimum_span=0.10,
            )
        )
        axis.set_title(label, fontsize=9)
        axis.grid(alpha=0.20)
        _set_compact_x_axis(axis, steps)
        axis.tick_params(axis="both", labelsize=7)
        count_axis.tick_params(axis="y", labelsize=7, colors=count_color)
        axis.tick_params(axis="y", colors=fraction_color)
        if column_index == 0:
            axis.set_ylabel("Document fraction", color=fraction_color, fontsize=8)
        if column_index == ncols - 1 or index == len(labels) - 1:
            count_axis.set_ylabel("Mean count", color=count_color, fontsize=8)

    for index in range(len(labels), nrows * ncols):
        row_index, column_index = divmod(index, ncols)
        axes[row_index][column_index].set_visible(False)

    from matplotlib.lines import Line2D

    figure.suptitle(title, fontsize=12)
    figure.supxlabel("Training step", fontsize=9, y=0.012)
    figure.legend(
        handles=[
            Line2D(
                [0], [0], color=fraction_color, marker="o", label="Document fraction"
            ),
            Line2D([0], [0], color=count_color, marker="s", label="Mean count"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.965),
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.93))
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def _plot_outputs(
    settings: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
    wide_rows: Sequence[Mapping[str, Any]],
    marker_rows: Sequence[Mapping[str, Any]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        print(
            f"[checkpoint behavior eval] matplotlib unavailable; skipping plots: {error}",
            file=sys.stderr,
        )
        return

    output_dir = Path(str(settings["output_dir"]))
    ordered = sorted(summaries, key=lambda item: int(item["checkpoint_step"]))
    steps = [int(item["checkpoint_step"]) for item in ordered]
    means = [float(item["rollout/mean_generated_tokens"]) for item in ordered]

    # The main length PNG intentionally contains one series only. Median/IQR
    # remain available in checkpoint_summary.csv without duplicating the plot.
    figure, axis = plt.subplots(figsize=(6.8, 4.4))
    axis.plot(steps, means, color="C0", marker="o", linewidth=2.2)
    axis.set_ylim(
        *_padded_axis_limits(means, lower_bound=0.0, minimum_span=20.0)
    )
    _set_compact_x_axis(axis, steps)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Mean rollout length (tokens)")
    axis.set_title("Mean rollout length across checkpoints")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "length_trends.png", dpi=170)
    plt.close(figure)

    checkpoint_names = [str(item["checkpoint"]) for item in ordered]
    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    trajectory_values: list[float] = []
    for row in wide_rows:
        values = [row.get(name) for name in checkpoint_names]
        if all(value is not None for value in values):
            axis.plot(steps, values, color="C0", alpha=0.08, linewidth=0.7)
            trajectory_values.extend(float(value) for value in values)
    if trajectory_values:
        axis.set_ylim(
            *_padded_axis_limits(
                trajectory_values, lower_bound=0.0, minimum_span=20.0
            )
        )
    _set_compact_x_axis(axis, steps)
    axis.set_xlabel("Training step")
    axis.set_ylabel("Rollout length (tokens)")
    axis.set_title("Per-problem rollout length trajectories")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "sample_length_trajectories.png", dpi=170)
    plt.close(figure)

    category_names = sorted(
        {
            key[len("behavior_occurrence/") : -len("_document_fraction")]
            for item in ordered
            for key in item
            if key.startswith("behavior_occurrence/")
            and key.endswith("_document_fraction")
        }
    )
    if category_names:
        category_series: dict[str, dict[int, tuple[float, float]]] = {}
        for category in category_names:
            category_series[category] = {
                int(item["checkpoint_step"]): (
                    float(
                        item.get(
                            f"behavior_occurrence/{category}_document_fraction", 0.0
                        )
                    ),
                    float(
                        item.get(f"behavior_occurrence/{category}_mean_count", 0.0)
                    ),
                )
                for item in ordered
            }
        category_labels = [category.replace("_", " ") for category in category_names]
        _plot_fraction_count_small_multiples(
            plt,
            output_path=output_dir / "behavior_category_trends.png",
            title="Behavior-category occurrence trends",
            steps=steps,
            series={
                label: category_series[category]
                for category, label in zip(
                    category_names, category_labels, strict=True
                )
            },
            labels=category_labels,
            columns=int(settings.get("plot_columns", 4)),
        )

    marker_series: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    for row in marker_rows:
        key = (str(row["category"]), str(row["marker"]))
        marker_series.setdefault(key, {})[int(row["checkpoint_step"])] = (
            float(row["document_fraction"]),
            float(row["mean_count"]),
        )

    def marker_change_score(key: tuple[str, str]) -> tuple[float, float, float]:
        values = marker_series[key].values()
        fractions = [value[0] for value in values]
        counts = [value[1] for value in values]
        fraction_range = max(fractions) - min(fractions)
        count_range = max(counts) - min(counts)
        normalized_count_range = count_range / max(max(counts), 1e-12)
        return (
            max(fraction_range, normalized_count_range),
            fraction_range,
            count_range,
        )

    ranked_markers = sorted(
        marker_series,
        key=marker_change_score,
        reverse=True,
    )[: int(settings.get("plot_top_markers", 12))]
    if ranked_markers:
        display_labels: list[str] = []
        display_series: dict[str, Mapping[int, tuple[float, float]]] = {}
        for category, marker in ranked_markers:
            marker_text = marker if len(marker) <= 24 else marker[:21] + "..."
            label = f"{category.replace('_', ' ')}\n{marker_text}"
            display_labels.append(label)
            display_series[label] = marker_series[(category, marker)]
        _plot_fraction_count_small_multiples(
            plt,
            output_path=output_dir / "behavior_marker_trends.png",
            title=(
                f"Top {len(ranked_markers)} most-changing behavior markers"
            ),
            steps=steps,
            series=display_series,
            labels=display_labels,
            columns=int(settings.get("plot_columns", 4)),
        )


def main() -> None:
    args = parse_args()
    settings = _resolve_settings(args)
    experiment_dir = Path(str(settings["experiment_dir"]))
    output_dir = Path(str(settings["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        _run_aggregate_only(settings, args)
        return

    checkpoints = _discover_checkpoints(args, experiment_dir)
    _infer_base_model(checkpoints, settings)

    try:
        import torch
        import transformers

        print(
            "[checkpoint behavior eval] runtime "
            f"torch={torch.__version__} transformers={transformers.__version__}"
        )
    except ImportError as error:
        raise RuntimeError(
            "torch, transformers, datasets, and PEFT (for LoRA) must be installed"
        ) from error

    samples, sample_manifest = _load_or_create_samples(
        settings,
        rebuild=bool(args.rebuild_sample_set),
        overwrite=bool(args.overwrite),
    )
    tokenizer, _, eos_info = _load_tokenizers_and_eos(settings)
    analyzer = RolloutBehaviorAnalyzer(
        tokenizer,
        extra_markers=settings.get("marker_dictionary"),
        focus_markers=settings.get("focus_markers"),
    )
    prepared_samples = _prepare_samples(tokenizer, samples, settings)
    evaluation_fingerprint = _evaluation_fingerprint(
        settings, sample_manifest, samples, eos_info
    )

    settings_to_save = dict(settings)
    settings_to_save.update(
        {
            "evaluation_fingerprint": evaluation_fingerprint,
            "checkpoints": [str(path) for path in checkpoints],
            "checkpoint_steps": [checkpoint_step(path.name) for path in checkpoints],
            "eos_token_ids": list(eos_info.token_ids),
            "eos_tokens": eos_info.token_strings,
            "eos_sources": eos_info.sources,
            "sample_set_path": str(output_dir / "selected_samples.jsonl"),
        }
    )
    _write_json(output_dir / "resolved_eval_config.json", settings_to_save)
    _write_json(
        output_dir / "behavior_marker_manifest.json",
        analyzer.manifest(eos_token_ids=eos_info.token_ids),
    )

    print("=" * 80)
    print(f"Experiment: {experiment_dir}")
    print(f"Base model: {settings['base_model']}")
    print(f"Checkpoints: {[path.name for path in checkpoints]}")
    print(f"Fixed samples: {len(samples)} ({settings['dataset_name']})")
    print(f"Prompt uses chat template: {settings['use_chat_template']}")
    print(f"System prompt: {settings['system_prompt']}")
    print(
        "Generation: "
        f"max_new_tokens={settings['max_new_tokens']} "
        f"do_sample={settings['do_sample']} "
        f"stop_after_boxed={settings['stop_after_boxed']}"
    )
    print(f"Output: {output_dir}")
    print("=" * 80, flush=True)

    checkpoint_records: dict[str, list[dict[str, Any]]] = {}
    summaries: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        records, summary = _evaluate_checkpoint(
            checkpoint,
            tokenizer,
            eos_info,
            analyzer,
            prepared_samples,
            settings,
            evaluation_fingerprint,
            overwrite=bool(args.overwrite),
        )
        checkpoint_records[checkpoint.name] = records
        summaries.append(summary)

    ordered_summaries, wide_rows, marker_rows = _write_aggregate_outputs(
        settings,
        checkpoint_records,
        summaries,
        marker_universe=_marker_universe(analyzer),
    )
    if args.plots:
        _plot_outputs(settings, ordered_summaries, wide_rows, marker_rows)
    print(
        f"[checkpoint behavior eval] all results written to {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
