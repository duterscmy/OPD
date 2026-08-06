#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import heapq
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Top-K OPD token debug JSONL")
    parser.add_argument("--input", required=True, help="JSONL path or glob")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-n", type=int, default=500)
    return parser.parse_args()


def new_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "overlap": 0.0,
        "reverse_kl": 0.0,
        "forward_kl": 0.0,
        "final_loss": 0.0,
        "student_target_logp": 0.0,
        "teacher_target_logp": 0.0,
        "logp_count": 0,
        "routes": defaultdict(int),
    }


def update(stats: dict[str, Any], token: dict[str, Any]) -> None:
    stats["count"] += 1
    stats["overlap"] += float(token["overlap"])
    stats["reverse_kl"] += float(token["reverse_kl"])
    stats["forward_kl"] += float(token["forward_kl"])
    stats["final_loss"] += float(token["final_token_loss"])
    stats["routes"][str(token["loss_route"])] += 1
    if "student_target_log_probability" in token:
        stats["student_target_logp"] += float(token["student_target_log_probability"])
        stats["teacher_target_logp"] += float(token["teacher_target_log_probability"])
        stats["logp_count"] += 1


def averaged(stats: dict[str, Any]) -> dict[str, Any]:
    count = max(int(stats["count"]), 1)
    logp_count = max(int(stats["logp_count"]), 1)
    return {
        "count": stats["count"],
        "mean_overlap": stats["overlap"] / count,
        "mean_reverse_kl": stats["reverse_kl"] / count,
        "mean_forward_kl": stats["forward_kl"] / count,
        "mean_final_loss": stats["final_loss"] / count,
        "mean_student_target_logp": stats["student_target_logp"] / logp_count,
        "mean_teacher_target_logp": stats["teacher_target_logp"] / logp_count,
        "routes": dict(stats["routes"]),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in sorted(glob.glob(args.input))]
    if not paths:
        raise FileNotFoundError(f"No files matched {args.input!r}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    token_stats: dict[tuple[int, str], dict[str, Any]] = defaultdict(new_stats)
    route_stats: dict[str, dict[str, Any]] = defaultdict(new_stats)
    overlap_stats: dict[str, dict[str, Any]] = defaultdict(new_stats)
    position_stats: dict[str, dict[str, Any]] = defaultdict(new_stats)
    highest_loss: list[tuple[float, int, dict[str, Any]]] = []
    samples = tokens = serial = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("event") != "topk_opd_sample_debug":
                    continue
                samples += 1
                for token in record.get("tokens", []):
                    tokens += 1
                    token_key = (int(token["target_token_id"]), str(token["target_token"]))
                    update(token_stats[token_key], token)
                    update(route_stats[str(token["loss_route"])], token)
                    overlap = min(max(float(token["overlap"]), 0.0), 1.0)
                    lower = min(int(overlap * 10), 9) / 10.0
                    overlap_key = f"[{lower:.1f},{lower + 0.1:.1f}]"
                    update(overlap_stats[overlap_key], token)
                    response_position = int(token["response_position"])
                    position_key = f"{((response_position - 1) // 50) * 50 + 1}-{((response_position - 1) // 50 + 1) * 50}"
                    update(position_stats[position_key], token)

                    serial += 1
                    detail = {
                        "source_file": str(path),
                        "line_number": line_number,
                        "global_step": record.get("global_step"),
                        "loss_mode": record.get("loss_mode"),
                        "response_position": response_position,
                        "target_token_id": token["target_token_id"],
                        "target_token": token["target_token"],
                        "loss_route": token["loss_route"],
                        "overlap": token["overlap"],
                        "reverse_kl": token["reverse_kl"],
                        "forward_kl": token["forward_kl"],
                        "final_token_loss": token["final_token_loss"],
                        "rollout_excerpt": str(record.get("student_rollout", ""))[:300],
                    }
                    item = (float(token["final_token_loss"]), serial, detail)
                    if len(highest_loss) < args.top_n:
                        heapq.heappush(highest_loss, item)
                    elif item[0] > highest_loss[0][0]:
                        heapq.heapreplace(highest_loss, item)

    token_rows = []
    for (token_id, token_text), stats in token_stats.items():
        row = {"token_id": token_id, "token": token_text, **averaged(stats)}
        row["routes"] = json.dumps(row["routes"], ensure_ascii=False, sort_keys=True)
        token_rows.append(row)
    token_rows.sort(key=lambda row: (-float(row["mean_final_loss"]), -int(row["count"])))

    route_rows = [{"loss_route": key, **averaged(value)} for key, value in route_stats.items()]
    overlap_rows = [{"overlap_bin": key, **averaged(value)} for key, value in overlap_stats.items()]
    position_rows = [{"position_bin": key, **averaged(value)} for key, value in position_stats.items()]
    for rows in (route_rows, overlap_rows, position_rows):
        for row in rows:
            row["routes"] = json.dumps(row["routes"], ensure_ascii=False, sort_keys=True)
    route_rows.sort(key=lambda row: row["loss_route"])
    overlap_rows.sort(key=lambda row: row["overlap_bin"])
    position_rows.sort(key=lambda row: int(str(row["position_bin"]).split("-")[0]))
    highest_rows = [item[2] for item in sorted(highest_loss, reverse=True)]

    metric_fields = [
        "count", "mean_overlap", "mean_reverse_kl", "mean_forward_kl",
        "mean_final_loss", "mean_student_target_logp", "mean_teacher_target_logp", "routes",
    ]
    write_csv(output_dir / "token_summary.csv", token_rows, ["token_id", "token", *metric_fields])
    write_csv(output_dir / "route_summary.csv", route_rows, ["loss_route", *metric_fields])
    write_csv(output_dir / "overlap_bin_summary.csv", overlap_rows, ["overlap_bin", *metric_fields])
    write_csv(output_dir / "position_bin_summary.csv", position_rows, ["position_bin", *metric_fields])
    write_csv(
        output_dir / "highest_loss_positions.csv",
        highest_rows,
        [
            "source_file", "line_number", "global_step", "loss_mode", "response_position",
            "target_token_id", "target_token", "loss_route", "overlap", "reverse_kl",
            "forward_kl", "final_token_loss", "rollout_excerpt",
        ],
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "input_files": [str(path) for path in paths],
                "sample_records": samples,
                "token_records": tokens,
                "unique_target_tokens": len(token_rows),
                "top_n": args.top_n,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Wrote analysis to {output_dir} ({samples} samples, {tokens} tokens)")


if __name__ == "__main__":
    main()
