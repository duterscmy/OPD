from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval_checkpoint_rollout_behavior import (
    StreamingBoxedDetector,
    _compact_step_ticks,
    _load_completed_checkpoint_outputs,
    _padded_axis_limits,
)
from opd.checkpoint_behavior_stats import (
    linear_slope,
    marker_summary_rows,
    overall_trend_summary,
    percentile,
    sample_length_outputs,
    summarize_checkpoint_records,
)


def _behavior(conclusion_count: int = 0) -> dict:
    conclusion_matches = (
        {
            "therefore": {
                "count": conclusion_count,
                "first_token_position": 2,
            }
        }
        if conclusion_count
        else {}
    )
    return {
        "token_count": 10,
        "categories": {
            "conclusion": {
                "count": conclusion_count,
                "document_hit": conclusion_count > 0,
                "first_token_position": 2 if conclusion_count else None,
                "first_relative_position": 0.2 if conclusion_count else None,
                "matched_markers": conclusion_matches,
            },
            "termination": {
                "count": 0,
                "document_hit": False,
                "first_token_position": None,
                "first_relative_position": None,
                "matched_markers": {},
            },
        },
        "repetition_continuation": {
            "eligible_position_fraction": 0.3,
            "actual_continuation_fraction": 0.1,
            "actual_given_eligible_fraction": 1.0 / 3.0,
        },
    }


def _record(sample_id: str, length: int, conclusion_count: int = 0) -> dict:
    return {
        "sample_id": sample_id,
        "dataset_index": int(sample_id[-1]),
        "prompt_length": 20,
        "rollout_length": length,
        "raw_rollout_length": length,
        "stop_reason": "boxed_answer",
        "emitted_eos": False,
        "hit_horizon": False,
        "raw_hit_horizon": False,
        "boxed_truncated": True,
        "appended_eos": True,
        "raw_boxed_count": 1,
        "raw_repeated_ngram_ratio": 0.1,
        "effective_repeated_ngram_ratio": 0.1,
        "student_behavior": _behavior(conclusion_count),
    }


class StreamingBoxedDetectorTests(unittest.TestCase):
    def test_marker_can_span_token_pieces(self) -> None:
        detector = StreamingBoxedDetector()
        self.assertFalse(detector.feed("answer: \\bo"))
        self.assertFalse(detector.feed("xed{"))
        self.assertFalse(detector.feed("32.5"))
        self.assertTrue(detector.feed("}"))

    def test_empty_placeholder_is_ignored(self) -> None:
        detector = StreamingBoxedDetector()
        self.assertFalse(detector.feed("instruction \\boxed{} then solve"))
        self.assertTrue(detector.feed(" result \\boxed{7}"))

    def test_nested_payload(self) -> None:
        detector = StreamingBoxedDetector()
        self.assertFalse(detector.feed("\\boxed{\\frac{1}"))
        self.assertTrue(detector.feed("{2}}"))


class CheckpointStatisticsTests(unittest.TestCase):
    def test_percentile_and_slope(self) -> None:
        self.assertEqual(percentile([0, 10], 0.25), 2.5)
        self.assertEqual(linear_slope([0, 25, 50], [10, 20, 30]), 0.4)

    def test_checkpoint_summary_uses_training_log_names(self) -> None:
        summary = summarize_checkpoint_records(
            "checkpoint-25",
            25,
            [_record("s0", 100, 1), _record("s1", 300, 0)],
        )
        self.assertEqual(summary["rollout/mean_generated_tokens"], 200.0)
        self.assertEqual(summary["rollout/median_generated_tokens"], 200.0)
        self.assertEqual(summary["rollout/boxed_answer_stop_fraction"], 1.0)
        self.assertEqual(
            summary["behavior_occurrence/conclusion_document_fraction"], 0.5
        )

    def test_cross_checkpoint_sample_outputs(self) -> None:
        records = {
            "checkpoint-25": [_record("s0", 100), _record("s1", 200)],
            "checkpoint-50": [_record("s0", 150), _record("s1", 180)],
        }
        steps = {"checkpoint-25": 25, "checkpoint-50": 50}
        wide, trends = sample_length_outputs(records, steps)
        self.assertEqual(wide[0]["checkpoint-50"], 150)
        by_id = {row["sample_id"]: row for row in trends}
        self.assertEqual(by_id["s0"]["delta_first_to_last"], 50.0)
        self.assertEqual(by_id["s1"]["delta_first_to_last"], -20.0)

    def test_marker_rows_include_zero_hit_checkpoint(self) -> None:
        records = {
            "checkpoint-25": [_record("s0", 100, 1)],
            "checkpoint-50": [_record("s0", 120, 0)],
        }
        rows = marker_summary_rows(
            records,
            {"checkpoint-25": 25, "checkpoint-50": 50},
            marker_universe=[("conclusion", "therefore")],
        )
        target = {
            (row["checkpoint"], row["marker"]): row["document_fraction"]
            for row in rows
        }
        self.assertEqual(target[("checkpoint-25", "therefore")], 1.0)
        self.assertEqual(target[("checkpoint-50", "therefore")], 0.0)

    def test_overall_trend(self) -> None:
        summaries = [
            {
                "checkpoint": "checkpoint-25",
                "checkpoint_step": 25,
                "rollout/mean_generated_tokens": 100.0,
            },
            {
                "checkpoint": "checkpoint-50",
                "checkpoint_step": 50,
                "rollout/mean_generated_tokens": 150.0,
            },
        ]
        sample_trends = [
            {"delta_first_to_last": 20.0},
            {"delta_first_to_last": -10.0},
        ]
        trend = overall_trend_summary(summaries, sample_trends)
        self.assertEqual(trend["mean_length_delta_first_to_last"], 50.0)
        self.assertEqual(trend["mean_length_slope_tokens_per_25_steps"], 50.0)
        self.assertEqual(trend["sample_fraction_increased"], 0.5)


class AggregateOnlyTests(unittest.TestCase):
    def test_adaptive_limits_and_compact_ticks(self) -> None:
        low, high = _padded_axis_limits([100.0, 110.0, 105.0], minimum_span=5.0)
        self.assertGreater(low, 90.0)
        self.assertLess(high, 120.0)
        self.assertEqual(
            _compact_step_ticks([25, 50, 75, 100, 125, 150, 175], 4),
            [25, 75, 125, 175],
        )

    def test_loads_completed_results_without_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            checkpoint_dir = output_dir / "checkpoints" / "checkpoint-25"
            checkpoint_dir.mkdir(parents=True)
            records = [_record("s0", 100)]
            with (checkpoint_dir / "rollouts.jsonl").open(
                "w", encoding="utf-8"
            ) as handle:
                for record in records:
                    handle.write(json.dumps(record) + "\n")
            summary = summarize_checkpoint_records(
                "checkpoint-25", 25, records
            )
            (checkpoint_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "summary": summary,
                    }
                ),
                encoding="utf-8",
            )
            loaded_records, loaded_summaries = (
                _load_completed_checkpoint_outputs(output_dir)
            )
            self.assertEqual(list(loaded_records), ["checkpoint-25"])
            self.assertEqual(loaded_records["checkpoint-25"][0]["sample_id"], "s0")
            self.assertEqual(
                loaded_summaries[0]["rollout/mean_generated_tokens"], 100.0
            )


if __name__ == "__main__":
    unittest.main()
