from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HorizonSchedule:
    """Rollout horizon for the two supported training strategies."""

    strategy: str
    prefix_length: int
    full_max_new_tokens: int

    def horizon(self, global_step: int) -> int:
        del global_step  # Kept in the signature for Trainer compatibility.
        if self.strategy == "full":
            return int(self.full_max_new_tokens)
        if self.strategy == "esr":
            return int(self.prefix_length)
        raise ValueError(
            f"Unsupported strategy={self.strategy!r}; expected 'full' or 'esr'."
        )
