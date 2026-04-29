"""Lazily-allocated ring buffer for audio samples.

Grows by doubling from a small initial allocation up to the requested capacity.
Once full, behaves as a standard ring buffer with zero further allocations.

Growth events for 60 min at 44100 Hz: ~7 doublings, worst single copy ~30ms.
Steady-state allocation for 60 min at 44100 Hz mono int16: 317 MB.
"""

import threading

import numpy as np

_INIT_SECONDS = 120  # initial alloc covers ~2 minutes


class RingBuffer:
    __slots__ = ("_buf", "_pos", "_full", "_capacity", "_total_writes",
                 "_lock", "_aux", "_aux_pos", "_aux_full")

    def __init__(self, capacity: int, sample_rate: int):
        init = min(_INIT_SECONDS * sample_rate, capacity)
        self._buf = np.zeros(init, dtype=np.int16)
        self._capacity = capacity
        self._pos = 0
        self._full = False
        self._total_writes = 0
        self._lock = threading.Lock()
        # Auxiliary buffer for a second audio source (e.g. system audio).
        # Allocated lazily on first aux write. Shares capacity with primary.
        self._aux = None
        self._aux_pos = 0
        self._aux_full = False

    def write(self, data: np.ndarray):
        n = len(data)
        with self._lock:
            if self._full:
                self._write_ring(data, n)
            else:
                self._write_growing(data, n)
            self._total_writes += n

    def write_aux(self, data: np.ndarray):
        """Write to the auxiliary (system audio) buffer.

        The aux buffer tracks its own write cursor but shares capacity
        with the primary buffer. Audio is mixed at read time.
        """
        n = len(data)
        with self._lock:
            if self._aux is None:
                self._aux = np.zeros(len(self._buf), dtype=np.int16)
                self._aux_pos = 0
                self._aux_full = False
            self._aux_write(data, n)

    def _aux_write(self, data: np.ndarray, n: int):
        if self._aux_full:
            self._aux_write_ring(data, n)
        else:
            self._aux_write_growing(data, n)

    def _aux_write_growing(self, data: np.ndarray, n: int):
        needed = self._aux_pos + n
        buf_len = len(self._aux)
        if needed > buf_len and buf_len < self._capacity:
            new_size = min(max(buf_len * 2, needed), self._capacity)
            new_buf = np.zeros(new_size, dtype=np.int16)
            new_buf[:self._aux_pos] = self._aux[:self._aux_pos]
            self._aux = new_buf

        if needed <= self._capacity:
            self._aux[self._aux_pos:needed] = data
            self._aux_pos = needed
            if self._aux_pos == self._capacity:
                self._aux_pos = 0
                self._aux_full = True
        else:
            fit = self._capacity - self._aux_pos
            self._aux[self._aux_pos:self._capacity] = data[:fit]
            self._aux[:n - fit] = data[fit:]
            self._aux_pos = n - fit
            self._aux_full = True

    def _aux_write_ring(self, data: np.ndarray, n: int):
        end = self._aux_pos + n
        if end <= self._capacity:
            self._aux[self._aux_pos:end] = data
        else:
            first = self._capacity - self._aux_pos
            self._aux[self._aux_pos:] = data[:first]
            self._aux[:n - first] = data[first:]
        self._aux_pos = end % self._capacity

    def _write_growing(self, data: np.ndarray, n: int):
        """Linear fill phase. Grows buffer by doubling when needed."""
        needed = self._pos + n
        buf_len = len(self._buf)
        if needed > buf_len and buf_len < self._capacity:
            new_size = min(max(buf_len * 2, needed), self._capacity)
            new_buf = np.zeros(new_size, dtype=np.int16)
            new_buf[: self._pos] = self._buf[: self._pos]
            self._buf = new_buf
            # Grow aux buffer to match if it exists
            if self._aux is not None and len(self._aux) < new_size:
                new_aux = np.zeros(new_size, dtype=np.int16)
                new_aux[:len(self._aux)] = self._aux
                self._aux = new_aux

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

    @staticmethod
    def mix_tracks(primary: np.ndarray, aux: np.ndarray) -> np.ndarray:
        """Mix two int16 tracks, clipping safely back to int16."""
        if len(aux) == 0:
            return primary
        mixed = primary.astype(np.int32) + aux.astype(np.int32)
        return np.clip(mixed, -32768, 32767).astype(np.int16)

    @staticmethod
    def _read_region(buf: np.ndarray, start_pos: int, n: int) -> np.ndarray:
        end_pos = start_pos + n
        if end_pos <= len(buf):
            return buf[start_pos:end_pos].copy()
        first = len(buf) - start_pos
        return np.concatenate([buf[start_pos:], buf[:n - first]])

    def _read_aux_track(self, start_pos: int, n: int) -> np.ndarray:
        if self._aux is None or n == 0:
            return np.zeros(n, dtype=np.int16)
        if start_pos >= len(self._aux):
            return np.zeros(n, dtype=np.int16)
        aux = self._read_region(self._aux, start_pos, n)
        if len(aux) < n:
            padded = np.zeros(n, dtype=np.int16)
            padded[:len(aux)] = aux
            return padded
        return aux

    def read_last_tracks(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Read the last n samples as aligned primary and auxiliary tracks."""
        with self._lock:
            avail = self.available
            n = min(n, avail)
            if n == 0:
                empty = np.array([], dtype=np.int16)
                return empty, empty
            if not self._full:
                start = self._pos - n
                primary = self._buf[start : self._pos].copy()
                return primary, self._read_aux_track(start, n)
            start = (self._pos - n) % self._capacity
            primary = self._read_region(self._buf, start, n)
            return primary, self._read_aux_track(start, n)

    def read_last(self, n: int) -> np.ndarray:
        """Read the last n samples as a mixed contiguous copy."""
        primary, aux = self.read_last_tracks(n)
        return self.mix_tracks(primary, aux)

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
        base = len(self._buf) * self._buf.itemsize
        if self._aux is not None:
            base += len(self._aux) * self._aux.itemsize
        return base

    def read_range_tracks(self, start_total: int, end_total: int) -> tuple[np.ndarray, np.ndarray]:
        """Read primary and auxiliary tracks between two write snapshots.

        Raises ValueError if the requested range has been overwritten.
        """
        with self._lock:
            now = self._total_writes
            if end_total > now:
                end_total = now
            n = end_total - start_total
            if n <= 0:
                empty = np.array([], dtype=np.int16)
                return empty, empty
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
                primary = self._buf[start_pos:end_pos].copy()
                return primary, self._read_aux_track(start_pos, n)
            primary = self._read_region(self._buf, start_pos, n)
            return primary, self._read_aux_track(start_pos, n)

    def read_range(self, start_total: int, end_total: int) -> np.ndarray:
        """Read samples between two total_writes snapshots as mixed mono.

        Raises ValueError if the requested range has been overwritten.
        """
        primary, aux = self.read_range_tracks(start_total, end_total)
        return self.mix_tracks(primary, aux)
