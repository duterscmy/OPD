from __future__ import annotations

import unittest


try:
    import torch

    from opd.behavior_probabilities import summarize_next_token_probabilities
except ModuleNotFoundError:
    torch = None
    summarize_next_token_probabilities = None


@unittest.skipIf(torch is None, "PyTorch is not installed in the lightweight test environment")
class BehaviorProbabilityTests(unittest.TestCase):
    def test_sparse_probability_and_terminal_probability(self) -> None:
        logits = torch.zeros((1, 4, 4), dtype=torch.float32)
        active = torch.tensor([[True, True, True, True]])
        targets = torch.tensor([[0, 1, 2, 3]])
        terminal = torch.zeros((1, 4), dtype=torch.float32)
        summary = summarize_next_token_probabilities(
            logits,
            active,
            {"marker/two_tokens": (0, 1)},
            targets=targets,
            terminal_logits=terminal,
        )
        metrics = summary["sets"]["marker/two_tokens"]
        self.assertAlmostEqual(metrics["mean"], 0.5, places=6)
        self.assertAlmostEqual(metrics["terminal_mean"], 0.5, places=6)
        self.assertAlmostEqual(metrics["target_fraction"], 0.5, places=6)

    def test_repetition_probability(self) -> None:
        completion = [[0, 1, 2, 3, 9, 0, 1, 2, 3]]
        logits = torch.zeros((1, len(completion[0]), 10), dtype=torch.float32)
        active = torch.ones((1, len(completion[0])), dtype=torch.bool)
        summary = summarize_next_token_probabilities(
            logits,
            active,
            {"marker/eos": (9,)},
            completion_ids=completion,
            repetition_ngram_size=4,
        )
        repetition = summary["repetition_continuation"]
        self.assertAlmostEqual(
            repetition["mean_probability_at_eligible_positions"],
            0.1,
            places=6,
        )
        self.assertGreater(repetition["actual_continuation_fraction"], 0.0)


if __name__ == "__main__":
    unittest.main()
