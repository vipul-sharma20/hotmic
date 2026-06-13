"""Voice activity detection via Silero VAD over onnxruntime.

Runs the silero_vad.onnx model (bundled with the silero-vad package) directly
with onnxruntime + numpy. The silero-vad Python API is deliberately not
imported: it pulls in torch (~400 MB RSS) to drive a 2 MB model. The
detection logic below is a faithful numpy port of silero's OnnxWrapper and
VADIterator, so probabilities and segment boundaries match the reference
implementation.

Two consumers:
- SpeechMonitor: streaming detection over the live ring buffer. Fed raw
  int16 samples at the capture rate, emits finished utterances as
  (start_total, end_total) ranges in RingBuffer.total_writes coordinates.
- has_speech: batch check on saved audio, used to skip transcribing silence.

Silero only accepts 16 kHz (512-sample windows), so capture audio is
linearly resampled. That introduces some aliasing above 8 kHz, which is
irrelevant for detection and acceptable for speech transcription.
"""

import importlib.util
import threading
from pathlib import Path

import numpy as np

VAD_RATE = 16000
_WINDOW = 512   # silero-vad requires exactly 512 samples per call at 16 kHz
_CONTEXT = 64   # the model wants the last 64 samples of the previous window

# Dedicated model for batch checks. The model carries RNN state, so it
# cannot be shared with a streaming SpeechMonitor or used concurrently.
_batch_model = None
_batch_lock = threading.Lock()


def _find_model_file() -> Path:
    # find_spec locates the package without executing it — importing
    # silero_vad would pull in torch, which is the whole thing we're avoiding.
    spec = importlib.util.find_spec("silero_vad")
    if spec and spec.submodule_search_locations:
        path = Path(spec.submodule_search_locations[0]) / "data" / "silero_vad.onnx"
        if path.exists():
            return path
    raise ImportError(
        "silero-vad is required for voice activity detection.\n"
        "Install it with: pip install -e '.[vad]'"
    )


class _OnnxVad:
    """Numpy port of silero-vad's OnnxWrapper (16 kHz mono, batch of 1)."""

    def __init__(self):
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "onnxruntime is required for voice activity detection.\n"
                "Install it with: pip install -e '.[vad]'"
            )
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(_find_model_file()),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._sr = np.array(VAD_RATE, dtype=np.int64)
        self.reset()

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, window: np.ndarray) -> float:
        """Speech probability for one float32 window of 512 samples."""
        x = np.concatenate([self._context, window.reshape(1, -1)], axis=1)
        prob, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = x[:, -_CONTEXT:]
        return prob.item()


def to_float_16k(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Convert int16 audio at sample_rate to float32 at 16 kHz."""
    f32 = audio.astype(np.float32) / 32768.0
    if sample_rate == VAD_RATE:
        return f32
    n_out = int(len(f32) * VAD_RATE / sample_rate)
    x = np.arange(n_out, dtype=np.float64) * (sample_rate / VAD_RATE)
    return np.interp(x, np.arange(len(f32)), f32).astype(np.float32)


def has_speech(audio: np.ndarray, sample_rate: int,
               min_speech_s: float = 0.3) -> bool:
    """Whether int16 audio contains at least min_speech_s of speech."""
    global _batch_model
    f32 = to_float_16k(audio, sample_rate)
    with _batch_lock:
        if _batch_model is None:
            _batch_model = _OnnxVad()
        _batch_model.reset()
        speech_windows = 0
        for i in range(0, len(f32) - _WINDOW + 1, _WINDOW):
            if _batch_model(f32[i:i + _WINDOW]) >= 0.5:
                speech_windows += 1
    return speech_windows * _WINDOW / VAD_RATE >= min_speech_s


class SpeechMonitor:
    """Streaming speech segmentation in ring-buffer coordinates.

    feed() consumes int16 samples at the capture rate and returns completed
    utterances as (start_total, end_total). Utterances longer than
    max_utterance_s are emitted in slices so live transcription keeps up
    during continuous speech.

    The start/end state machine (threshold with -0.15 release hysteresis,
    min-silence wait, symmetric padding) mirrors silero's VADIterator.
    """

    def __init__(self, sample_rate: int, start_total: int,
                 threshold: float = 0.5, min_silence_ms: int = 800,
                 speech_pad_ms: int = 200, max_utterance_s: float = 30.0):
        self._model = _OnnxVad()
        self._sr = sample_rate
        self._origin = start_total
        self._threshold = threshold
        self._min_silence = int(min_silence_ms * VAD_RATE / 1000)
        self._pad = int(speech_pad_ms * VAD_RATE / 1000)
        self._max_utt = int(max_utterance_s * VAD_RATE)
        self._in_total = 0    # capture-rate samples consumed
        self._out_total = 0   # 16 kHz samples produced by the resampler
        self._fed = 0         # 16 kHz samples fed to the VAD
        self._pending = np.empty(0, dtype=np.float32)
        self._temp_end = 0      # 16 kHz position where current silence began
        self._utt_start = None  # 16 kHz position where current utterance began
        self.speaking = False

    def rebase(self, start_total: int):
        """Pin the ring position of the first sample that will be fed.

        Model load in __init__ takes time; callers that construct while
        audio keeps flowing must rebase to the ring cursor right before the
        first feed() or every emitted range is shifted by the load time.
        """
        if self._in_total:
            raise RuntimeError("rebase() is only valid before feed()")
        self._origin = start_total

    def _to_total(self, pos_16k: int) -> int:
        return self._origin + pos_16k * self._sr // VAD_RATE

    def _resample(self, samples: np.ndarray) -> np.ndarray:
        f32 = samples.astype(np.float32) / 32768.0
        if self._sr == VAD_RATE:
            self._in_total += len(samples)
            self._out_total += len(samples)
            return f32
        # Cumulative bookkeeping keeps capture/16k clocks drift-free across
        # chunk boundaries (the per-chunk rounding never accumulates).
        chunk_start = self._in_total
        self._in_total += len(samples)
        out_end = self._in_total * VAD_RATE // self._sr
        x = (np.arange(self._out_total, out_end, dtype=np.float64)
             * (self._sr / VAD_RATE)) - chunk_start
        self._out_total = out_end
        return np.interp(x, np.arange(len(f32)), f32).astype(np.float32)

    def _step(self, prob: float) -> dict | None:
        """One window of silero's VADIterator state machine."""
        if prob >= self._threshold and self._temp_end:
            self._temp_end = 0
        if prob >= self._threshold and not self.speaking:
            return {"start": max(0, self._fed - self._pad - _WINDOW)}
        if prob < self._threshold - 0.15 and self.speaking:
            if not self._temp_end:
                self._temp_end = self._fed
            if self._fed - self._temp_end >= self._min_silence:
                end = self._temp_end + self._pad - _WINDOW
                self._temp_end = 0
                return {"end": end}
        return None

    def feed(self, samples: np.ndarray) -> list[tuple[int, int]]:
        """Consume new samples, return any completed utterance ranges."""
        buf = np.concatenate([self._pending, self._resample(samples)])
        events: list[tuple[int, int]] = []
        offset = 0
        while offset + _WINDOW <= len(buf):
            prob = self._model(buf[offset:offset + _WINDOW])
            offset += _WINDOW
            self._fed += _WINDOW
            result = self._step(prob)
            if result and "start" in result:
                self._utt_start = result["start"]
                self.speaking = True
            elif result and "end" in result:
                # end can precede _utt_start if a force-flush slice happened
                # during the silence run-up; skip the inverted remainder.
                if self._utt_start is not None:
                    end = min(result["end"], self._fed)
                    if end > self._utt_start:
                        events.append((self._to_total(self._utt_start),
                                       self._to_total(end)))
                self._utt_start = None
                self.speaking = False
            elif self.speaking and self._fed - self._utt_start >= self._max_utt:
                events.append((self._to_total(self._utt_start),
                               self._to_total(self._fed)))
                self._utt_start = self._fed
        self._pending = buf[offset:]
        return events

    def flush(self) -> list[tuple[int, int]]:
        """Close out an in-progress utterance (e.g. on shutdown)."""
        if not self.speaking or self._utt_start is None:
            return []
        event = (self._to_total(self._utt_start), self._to_total(self._fed))
        self._utt_start = None
        self.speaking = False
        return [event] if event[1] > event[0] else []
