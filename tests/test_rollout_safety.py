from __future__ import annotations

import unittest

from opd.rollout_safety import (
    finalize_math_completion,
    first_complete_boxed_line_end,
    repeated_ngram_ratio,
    resolve_rollout_eos,
    truncate_after_first_boxed,
    truncate_completion,
)

try:
    import torch

    from opd.adaptive_kl_losses import _masked_sequence_mean
except ModuleNotFoundError:
    torch = None
    _masked_sequence_mean = None


class FakeTokenizer:
    def __init__(self, vocab: dict[str, int], eos_token_id: int, unk_token_id: int | None = None):
        self.vocab = dict(vocab)
        self.inverse = {value: key for key, value in vocab.items()}
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

    def convert_ids_to_tokens(self, token_id: int) -> str | None:
        return self.inverse.get(int(token_id))

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token in self.vocab:
            return self.vocab[token]
        return self.unk_token_id

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(self.inverse.get(int(token_id), "") for token_id in token_ids)


class RolloutSafetyTest(unittest.TestCase):
    def setUp(self) -> None:
        vocab = {
            "<|endoftext|>": 0,
            "a": 1,
            "<|im_end|>": 2,
            "b": 3,
        }
        self.student = FakeTokenizer(vocab, eos_token_id=0)
        self.teacher = FakeTokenizer(vocab, eos_token_id=2)
        self.eos_info = resolve_rollout_eos(
            self.student,
            self.teacher,
            student_generation_eos=0,
            teacher_generation_eos=[2, 0],
        )

    def test_teacher_eos_is_mapped_into_student_vocabulary(self) -> None:
        self.assertEqual(self.eos_info.token_ids, (0, 2))
        self.assertEqual(self.eos_info.stop_reason(0), "student_eos")
        self.assertEqual(self.eos_info.stop_reason(2), "teacher_eos")
        self.assertEqual(self.eos_info.preferred_teacher_eos_id(), 2)

    def test_teacher_eos_truncates_before_student_padding(self) -> None:
        result = truncate_completion(
            [1, 3, 2, 0, 0],
            eos_info=self.eos_info,
            pad_token_id=0,
            horizon=5,
        )
        self.assertEqual(result.token_ids, [1, 3, 2])
        self.assertEqual(result.stop_reason, "teacher_eos")
        self.assertFalse(result.hit_horizon)

    def test_pad_equal_to_student_eos_keeps_real_eos(self) -> None:
        result = truncate_completion(
            [1, 3, 0, 0],
            eos_info=self.eos_info,
            pad_token_id=0,
            horizon=4,
        )
        self.assertEqual(result.token_ids, [1, 3, 0])
        self.assertTrue(result.emitted_eos)
        self.assertEqual(result.stop_reason, "student_eos")

    def test_no_eos_at_horizon_is_marked_truncated(self) -> None:
        result = truncate_completion(
            [1, 3, 1, 3, 1],
            eos_info=self.eos_info,
            pad_token_id=0,
            horizon=4,
        )
        self.assertEqual(result.token_ids, [1, 3, 1, 3])
        self.assertTrue(result.hit_horizon)
        self.assertEqual(result.stop_reason, "max_length")

    def test_repeated_ngram_ratio(self) -> None:
        self.assertGreater(repeated_ngram_ratio([1, 2, 1, 2, 1, 2], n=2), 0.0)
        self.assertEqual(repeated_ngram_ratio([1, 2, 3, 4], n=2), 0.0)

    def test_complete_boxed_answer_line_is_detected(self) -> None:
        text = "reasoning\n\\boxed{\\frac{1}{2}}.\nmore"
        end = first_complete_boxed_line_end(text)
        self.assertEqual(text[:end], "reasoning\n\\boxed{\\frac{1}{2}}.\n")

    def test_empty_boxed_placeholder_is_ignored(self) -> None:
        text = "use \\boxed{} here\nwork\n\\boxed{7}\nrepeat"
        end = first_complete_boxed_line_end(text)
        self.assertEqual(text[:end], "use \\boxed{} here\nwork\n\\boxed{7}\n")

    def test_repeated_boxed_tail_is_removed(self) -> None:
        tokenizer = FakeTokenizer(
            {
                "work\n": 10,
                "\\boxed{1}\n": 11,
                "\\boxed{1}\nagain": 12,
            },
            eos_token_id=0,
        )
        trimmed, did_trim, boxed_count = truncate_after_first_boxed(
            tokenizer, [10, 11, 12]
        )
        self.assertEqual(trimmed, [10, 11])
        self.assertTrue(did_trim)
        self.assertEqual(boxed_count, 2)

    def test_math_boundary_appends_teacher_eos_without_claiming_it_was_emitted(self) -> None:
        tokenizer = FakeTokenizer(
            {
                "work\n": 10,
                "\\boxed{1}\n": 11,
                "\\boxed{1}\nagain": 12,
                "<|im_end|>": 2,
            },
            eos_token_id=0,
        )
        raw = truncate_completion(
            [10, 11, 12],
            eos_info=self.eos_info,
            pad_token_id=0,
            horizon=3,
        )
        result = finalize_math_completion(
            raw,
            tokenizer,
            repetition_ngram_size=2,
            truncate_after_boxed_answer=True,
            append_eos_after_boxed_answer=True,
            terminal_eos_token_id=2,
        )
        self.assertEqual(result.token_ids, [10, 11, 2])
        self.assertEqual(result.stop_reason, "boxed_answer")
        self.assertTrue(result.raw_hit_horizon)
        self.assertFalse(result.hit_horizon)
        self.assertTrue(result.boxed_truncated)
        self.assertTrue(result.appended_eos)
        self.assertFalse(result.emitted_eos)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_sequence_normalization_is_length_neutral(self) -> None:
        values = torch.tensor([[1.0, 1.0, 0.0], [10.0, 0.0, 0.0]])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        loss, per_sequence = _masked_sequence_mean(values, mask)
        self.assertTrue(torch.allclose(per_sequence, torch.tensor([1.0, 10.0])))
        self.assertAlmostEqual(float(loss), 5.5)

    @unittest.skipIf(torch is None, "PyTorch is not installed")
    def test_zero_weight_drops_truncated_sequence(self) -> None:
        values = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
        mask = torch.ones_like(values, dtype=torch.bool)
        loss, _ = _masked_sequence_mean(
            values,
            mask,
            sequence_weights=torch.tensor([1.0, 0.0]),
        )
        self.assertAlmostEqual(float(loss), 1.0)


if __name__ == "__main__":
    unittest.main()
