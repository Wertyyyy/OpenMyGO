from typing import List, Sequence, Tuple

import numpy as np
from numba import njit


U64_MASK = np.uint64((1 << 64) - 1)
MASK_INT = (1 << 64) - 1
BASE1_INT = 1_000_000_007
BASE2_INT = 1_000_000_009


@njit(cache=True)
def _hash_range(
    h: np.ndarray,
    p: np.ndarray,
    left: int,
    right: int,
) -> int:
    return int((h[right] - (h[left] * p[right - left])) & U64_MASK)


@njit(cache=True)
def _equal_substring(
    h1: np.ndarray,
    h2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    left_a: int,
    left_b: int,
    length: int,
) -> bool:
    right_a = left_a + length
    right_b = left_b + length

    a1 = _hash_range(h1, p1, left_a, right_a)
    b1 = _hash_range(h1, p1, left_b, right_b)
    if a1 != b1:
        return False

    a2 = _hash_range(h2, p2, left_a, right_a)
    b2 = _hash_range(h2, p2, left_b, right_b)
    return a2 == b2


@njit(cache=True)
def _find_previous_match(
    h1: np.ndarray,
    h2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    tail_start: int,
    current_start: int,
    pattern_len: int,
    max_gap: int,
) -> int:
    right = current_start - pattern_len
    if right < 0:
        return -1

    left = max(0, current_start - pattern_len - max_gap)
    for candidate_start in range(right, left - 1, -1):
        if _equal_substring(h1, h2, p1, p2, candidate_start, tail_start, pattern_len):
            return candidate_start
    return -1


@njit(cache=True)
def _resolve_gap(pattern_len: int, allow_gap: bool, max_gap_ratio: float) -> int:
    if not allow_gap:
        return 0
    # Requested formula: int(pattern_len * ratio - 1)
    gap = int(pattern_len * max_gap_ratio - 1.0)
    if gap < 0:
        return 0
    return gap


@njit(cache=True)
def _detect_core(
    h1: np.ndarray,
    h2: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    n: int,
    min_repeat_tokens: int,
    min_repeat_count: int,
    allow_gap: bool,
    max_gap_ratio: float,
) -> Tuple[int, int, int, int, int]:
    # Returns: found(0/1), pattern_len, repeat_count, total_repeat_tokens, max_gap_used
    if n < min_repeat_tokens:
        return 0, 0, 0, 0, 0

    required_count_floor = min_repeat_count + 1
    max_pattern_len = min(n // 2, n // required_count_floor)
    if max_pattern_len <= 0:
        return 0, 0, 0, 0, 0

    for pattern_len in range(1, max_pattern_len + 1):
        required_by_tokens = (min_repeat_tokens // pattern_len) + 1
        required = max(required_count_floor, required_by_tokens)
        if required * pattern_len > n:
            continue

        tail_start = n - pattern_len
        repeat_count = 1
        current_start = tail_start
        max_gap = _resolve_gap(pattern_len, allow_gap, max_gap_ratio)

        while True:
            max_additional_matches = current_start // pattern_len
            if repeat_count + max_additional_matches < required:
                break

            prev_start = _find_previous_match(
                h1,
                h2,
                p1,
                p2,
                tail_start,
                current_start,
                pattern_len,
                max_gap,
            )
            if prev_start < 0:
                break

            repeat_count += 1
            if repeat_count >= required:
                break
            current_start = prev_start

        total_repeat_tokens = repeat_count * pattern_len
        if total_repeat_tokens > min_repeat_tokens and repeat_count > min_repeat_count:
            return 1, pattern_len, repeat_count, total_repeat_tokens, max_gap

    return 0, 0, 0, 0, 0


class IncrementalTokenRepetitionDetector:
    """Incremental tail repetition detector for token id sequences (numba-jit core)."""

    def __init__(
        self,
        min_repeat_tokens: int = 120,
        min_repeat_count: int = 30,
        allow_gap: bool = True,
        max_gap_ratio: float = 1.0,
    ) -> None:
        if min_repeat_tokens < 0:
            raise ValueError("min_repeat_tokens must be >= 0")
        if min_repeat_count < 1:
            raise ValueError("min_repeat_count must be >= 1")
        if max_gap_ratio < 0:
            raise ValueError("max_gap_ratio must be >= 0")

        self.min_repeat_tokens = min_repeat_tokens
        self.min_repeat_count = min_repeat_count
        self.allow_gap = allow_gap
        self.max_gap_ratio = max_gap_ratio

        self._tokens: List[int] = []
        self._n = 0

        self._h1 = [0]
        self._h2 = [0]
        self._p1 = [1]
        self._p2 = [1]

        self._h1_np: np.ndarray | None = None
        self._h2_np: np.ndarray | None = None
        self._p1_np: np.ndarray | None = None
        self._p2_np: np.ndarray | None = None
        self._numpy_dirty = True

    @property
    def length_tokens(self) -> int:
        return self._n

    def append(self, token_ids: Sequence[int]) -> None:
        if not token_ids:
            return

        self._tokens.extend(token_ids)

        h1 = self._h1[-1]
        h2 = self._h2[-1]
        p1 = self._p1[-1]
        p2 = self._p2[-1]

        for token in token_ids:
            code = int(token) + 1
            h1 = ((h1 * BASE1_INT) + code) & MASK_INT
            h2 = ((h2 * BASE2_INT) + code) & MASK_INT
            p1 = (p1 * BASE1_INT) & MASK_INT
            p2 = (p2 * BASE2_INT) & MASK_INT
            self._h1.append(h1)
            self._h2.append(h2)
            self._p1.append(p1)
            self._p2.append(p2)

        self._n += len(token_ids)
        self._numpy_dirty = True

    def append_and_detect(self, token_ids: Sequence[int]) -> Tuple[bool, str]:
        self.append(token_ids)
        return self.detect()

    def reset(self) -> None:
        self._tokens.clear()
        self._n = 0
        self._h1 = [0]
        self._h2 = [0]
        self._p1 = [1]
        self._p2 = [1]
        self._h1_np = None
        self._h2_np = None
        self._p1_np = None
        self._p2_np = None
        self._numpy_dirty = True

    def detect(self) -> Tuple[bool, str]:
        n = self._n
        if n < self.min_repeat_tokens:
            return False, ""

        self._ensure_numpy_cache()
        assert self._h1_np is not None
        assert self._h2_np is not None
        assert self._p1_np is not None
        assert self._p2_np is not None

        found, pattern_len, repeat_count, total_repeat_tokens, max_gap_used = _detect_core(
            self._h1_np,
            self._h2_np,
            self._p1_np,
            self._p2_np,
            n,
            self.min_repeat_tokens,
            self.min_repeat_count,
            self.allow_gap,
            self.max_gap_ratio,
        )

        if not found:
            return False, ""

        tail_start = n - pattern_len
        pattern = self._pattern_preview(tail_start, pattern_len)
        reason = (
            "token_repetition_detected: "
            f"pattern={pattern}, "
            f"pattern_length={pattern_len}, "
            f"repeat_count={repeat_count}, "
            f"total_repeat_tokens={total_repeat_tokens}, "
            f"min_repeat_tokens={self.min_repeat_tokens}, "
            f"min_repeat_count={self.min_repeat_count}, "
            f"allow_gap={self.allow_gap}, "
            f"max_gap_ratio={self.max_gap_ratio}, "
            f"max_gap={max_gap_used}"
        )
        return True, reason

    def _ensure_numpy_cache(self) -> None:
        if not self._numpy_dirty:
            return
        self._h1_np = np.asarray(self._h1, dtype=np.uint64)
        self._h2_np = np.asarray(self._h2, dtype=np.uint64)
        self._p1_np = np.asarray(self._p1, dtype=np.uint64)
        self._p2_np = np.asarray(self._p2, dtype=np.uint64)
        self._numpy_dirty = False

    def _pattern_preview(self, start: int, length: int) -> str:
        pattern = self._tokens[start : start + length]
        if len(pattern) <= 80:
            return repr(pattern)
        return repr(pattern[:77] + "...")
