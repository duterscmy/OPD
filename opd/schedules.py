from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HorizonSchedule:
    strategy: str
    prefix_length: int
    full_max_new_tokens: int
    curriculum_lengths: list[int]
    curriculum_boundaries: list[int]
    reflection_rollout_max_tokens: int
    correctness_rollout_max_tokens: int

    def horizon(self, global_step: int) -> int:
        if self.strategy == "full":
            return int(self.full_max_new_tokens)
        if self.strategy == "esr":
            return int(self.prefix_length)
        if self.strategy == "reflection":
            return int(self.reflection_rollout_max_tokens)
        if self.strategy == "correctness_esr":
            return int(self.correctness_rollout_max_tokens)
        if self.strategy == "curriculum":
            if not self.curriculum_lengths:
                return int(self.prefix_length)
            chosen = self.curriculum_lengths[0]
            for boundary, length in zip(self.curriculum_boundaries, self.curriculum_lengths, strict=False):
                if global_step >= int(boundary):
                    chosen = int(length)
            return int(chosen)
        raise ValueError(f"Unknown strategy: {self.strategy}")
