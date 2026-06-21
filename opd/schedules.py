from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HorizonSchedule:
    strategy: str
    prefix_length: int
    full_max_new_tokens: int
    curriculum_lengths: list[int]
    curriculum_boundaries: list[int]
    reflection_rollout_max_tokens: int

    def horizon(self, global_step: int) -> int:
        if self.strategy == "full":
            return self.full_max_new_tokens
        if self.strategy == "esr":
            return self.prefix_length
        if self.strategy == "reflection":
            return self.reflection_rollout_max_tokens
        if self.strategy == "curriculum":
            idx = 0
            for i, boundary in enumerate(self.curriculum_boundaries):
                if global_step >= boundary:
                    idx = i
                else:
                    break
            return self.curriculum_lengths[idx]
        raise ValueError(f"Unknown strategy: {self.strategy}")
