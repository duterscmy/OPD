from __future__ import annotations

import unittest
from pathlib import Path

from opd.config import (
    infer_effective_lengths,
    load_config_with_base,
    validate_experiment_config,
)


class BehaviorConfigTests(unittest.TestCase):
    def test_current_reverse_kl_config_enables_compatible_monitoring(self) -> None:
        config_path = (
            Path(__file__).parents[1] / "configs" / "math_topk_reverse.yaml"
        )
        cfg = load_config_with_base(str(config_path), [])
        validate_experiment_config(cfg)
        infer_effective_lengths(cfg)
        self.assertTrue(cfg["behavior_monitor_enabled"])
        self.assertTrue(cfg["behavior_probe_enabled"])
        self.assertIsNone(cfg["behavior_monitor_every_n_loss_calls"])
        self.assertEqual(cfg["effective_max_new_tokens"], 3072)
        self.assertFalse(cfg["rollout_append_eos_after_boxed_answer"])


if __name__ == "__main__":
    unittest.main()
