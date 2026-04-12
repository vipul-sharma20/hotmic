"""Lazily-allocated ring buffer for audio samples.

Grows by doubling from a small initial allocation up to the requested capacity.
Once full, behaves as a standard ring buffer with zero further allocations.

Growth events for 60 min at 44100 Hz: ~7 doublings, worst single copy ~30ms.
Steady-state allocation for 60 min at 44100 Hz mono int16: 317 MB.
"""

import numpy as np

_INIT_SECONDS = 120  # initial alloc covers ~2 minutes


class RingBuffer:
    __slots__ = ("_buf", "_pos", "_full", "_capacity", "_total_writes")

    def __init__(self, capacity: int, sample_rate: int):
        init = min(_INIT_SECONDS * sample_rate, capacity)
        self._buf = np.zeros(init, dtype=np.int16)
        self._capacity = capacity
        self._pos = 0
        self._full = False
        self._total_writes = 0

    def write(self, data: np.ndarray):
        n = len(data)
        if self._full:
            self._write_ring(data, n)
        else:
            self._write_growing(data, n)
        self._total_writes += n

    def _write_growing(self, data: np.ndarray, n: int):
        """Linear fill phase. Grows buffer by doubling when needed."""
        needed = self._pos + n
        buf_len = len(self._buf)
        if needed > buf_len and buf_len < self._capacity:
            new_size = min(max(buf_len * 2, needed), self._capacity)
            new_buf = np.zeros(new_size, dtype=np.int16)
            new_buf[: self._pos] = self._buf[: self._pos]
            self._buf = new_buf

        if needed <= self._capacity:
            self._buf[self._pos : needed] = data
            self._pos = needed
            if self._pos == self._capacity:
                self._pos = 0
                self._full = True
        else:
            # This write crosses capacity — transition to ring phase
            fit = self._capacity - self._pos
            self._buf[self._pos : self._capacity] = data[:fit]
            self._buf[: n - fit] = data[fit:]
            self._pos = n - fit
            self._full = True

    def _write_ring(self, data: np.ndarray, n: int):
        """Wrap-around phase. No allocations."""
        end = self._pos + n
        if end <= self._capacity:
            self._buf[self._pos : end] = data
        else:
            first = self._capacity - self._pos
            self._buf[self._pos :] = data[:first]
            self._buf[: n - first] = data[first:]
        self._pos = end % self._capacity

    def read_last(self, n: int) -> np.ndarray:
        """Read the last n samples as a contiguous copy."""
        avail = self.available
        n = min(n, avail)
        if n == 0:
            return np.array([], dtype=np.int16)
        if not self._full:
            return self._buf[self._pos - n : self._pos].copy()
        start = (self._pos - n) % self._capacity
        if start < self._pos:
            return self._buf[start : self._pos].copy()
        return np.concatenate([self._buf[start : self._capacity], self._buf[: self._pos]])

    @property
    def available(self) -> int:
        """Number of valid samples in the buffer."""
        return self._capacity if self._full else self._pos

    @property
    def pos(self) -> int:
        """Current write cursor position in the buffer."""
        return self._pos

    @property
    def total_writes(self) -> int:
        """Monotonic count of all samples written since creation."""
        return self._total_writes

    @property
    def allocated_bytes(self) -> int:
        """Current memory usage of the backing array."""
        return len(self._buf) * self._buf.itemsize

    def read_range(self, start_total: int, end_total: int) -> np.ndarray:
        """Read samples between two total_writes snapshots.

        Raises ValueError if the requested range has been overwritten.
        """
        now = self._total_writes
        if end_total > now:
            end_total = now
        n = end_total - start_total
        if n <= 0:
            return np.array([], dtype=np.int16)
        # Check both endpoints are still in the buffer
        oldest_available = now - self.available
        if start_total < oldest_available:
            raise ValueError(
                f"Mark audio overwritten: needed sample {start_total}, "
                f"oldest available is {oldest_available}"
            )
        # Convert totals to buffer positions
        start_pos = start_total % self._capacity
        end_pos = end_total % self._capacity
        if not self._full:
            return self._buf[start_pos:end_pos].copy()
        if start_pos < end_pos:
            return self._buf[start_pos:end_pos].copy()
        return np.concatenate([self._buf[start_pos:self._capacity], self._buf[:end_pos]])
