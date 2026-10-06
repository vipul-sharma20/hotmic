"""Transcribe audio with mlx-whisper or OpenRouter, with optional speaker diarization.

The transcription backend is chosen by ``HOTMIC_TRANSCRIBE_BACKEND``: ``mlx``
(default) runs whisper on-device; ``openrouter`` uploads 16 kHz mono audio to
OpenRouter's speech-to-text endpoint (model from ``HOTMIC_OPENROUTER_MODEL``,
key from ``OPENROUTER_API_KEY``), so the MLX model never loads.

Diarization is offloaded to a remote pyannote service when
``HOTMIC_DIARIZE_URL`` is set. The request runs concurrently with local
MLX transcription and is merged at the end, so it never sits on the recording
or live-transcription path. If the service is unreachable the transcript is
still written, just without speaker labels — there is no local fallback once a
remote backend is configured.
"""

import base64
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

import numpy as np

from . import __version__

_MODEL = "mlx-community/whisper-large-v3-turbo"

_BACKEND = "HOTMIC_TRANSCRIBE_BACKEND"
_OPENROUTER_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
_OPENROUTER_MODEL = "HOTMIC_OPENROUTER_MODEL"
_OPENROUTER_DEFAULT_MODEL = "openai/whisper-large-v3-turbo"
_OPENROUTER_TIMEOUT = 120
_OPENROUTER_RETRIES = 3
# 5 min of 16 kHz mono int16 is ~9.6 MB (~13 MB base64): under OpenRouter's
# 25 MB upload cap and well inside its 60 s per-request provider timeout.
_CHUNK_S = 300

_DIARIZE_URL = "HOTMIC_DIARIZE_URL"
_DIARIZE_TOKEN = "HOTMIC_DIARIZE_TOKEN"
_DIARIZE_TIMEOUT = float(os.environ.get("HOTMIC_DIARIZE_TIMEOUT", "600"))


def _release_mlx_cache():
    """Return MLX's Metal buffer cache to the OS.

    The cache grows to peak-activation size on every transcription and never
    shrinks on its own — multi-GB over a long session for a daemon that
    transcribes in bursts. Clearing between utterances trades a little buffer
    reuse for a flat memory profile.
    """
    import mlx.core as mx
    mx.clear_cache()


def _backend() -> str:
    name = os.environ.get(_BACKEND, "").strip().lower() or "mlx"
    if name not in ("mlx", "openrouter"):
        raise RuntimeError(f"Unknown {_BACKEND}={name!r}; use 'mlx' or 'openrouter'.")
    if name == "openrouter" and not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(f"{_BACKEND}=openrouter requires OPENROUTER_API_KEY.")
    return name


def _openrouter_model() -> str:
    return os.environ.get(_OPENROUTER_MODEL) or _OPENROUTER_DEFAULT_MODEL


def backend_label() -> str:
    """Validate the configured backend and describe it, e.g. for the banner."""
    if _backend() == "openrouter":
        return f"openrouter {_openrouter_model()}"
    return f"mlx {_MODEL.split('/')[-1]}"


def transcribe_wav(wav_path: Path, diarize: bool = False) -> tuple[Path, Path]:
    """Transcribe a WAV file, writing .txt and .srt alongside it.

    If diarize=True, runs speaker diarization and labels each
    segment with a speaker ID.

    Returns (txt_path, srt_path).
    """
    backend = _backend()
    if backend == "mlx":
        try:
            import mlx_whisper
        except ImportError:
            raise SystemExit(
                "mlx-whisper is required for transcription.\n"
                "Install it with: pip install -e '.[transcribe]'"
            )

    # Kick off diarization (remote GPU or local lib) before transcribing, so
    # it overlaps transcription instead of running after it.
    diar = {}
    diar_thread = None
    if diarize:
        diar_thread = threading.Thread(
            target=lambda: diar.update(segments=_diarize_segments(wav_path)),
            daemon=True,
        )
        diar_thread.start()

    if backend == "openrouter":
        segments = _openrouter_transcribe_wav(wav_path)
    else:
        result = mlx_whisper.transcribe(str(wav_path), path_or_hf_repo=_MODEL)
        _release_mlx_cache()
        segments = result.get("segments", [])

    if diar_thread is not None:
        diar_thread.join()
        speaker_segments = diar.get("segments")
        if speaker_segments:
            for seg in segments:
                mid = (seg["start"] + seg["end"]) / 2
                seg["speaker"] = _find_speaker(speaker_segments, mid)
        else:
            print("Diarization unavailable; writing transcript without speaker labels.")

    txt_path = wav_path.with_suffix(".txt")
    srt_path = wav_path.with_suffix(".srt")

    txt_path.write_text(_format_txt(segments))
    srt_path.write_text(_format_srt(segments))

    return txt_path, srt_path


def transcribe_audio(audio_16k) -> str:
    """Transcribe a float32 16 kHz mono array, returning plain text.

    Used by live transcription, where utterances come straight from the
    ring buffer and never touch disk.
    """
    if _backend() == "openrouter":
        pcm = (np.clip(audio_16k, -1.0, 1.0) * 32767).astype(np.int16)
        return _openrouter_request(_wav_bytes(pcm), verbose=False).get("text", "").strip()

    try:
        import mlx_whisper
    except ImportError:
        raise ImportError(
            "mlx-whisper is required for transcription.\n"
            "Install it with: pip install -e '.[transcribe]'"
        )

    result = mlx_whisper.transcribe(audio_16k, path_or_hf_repo=_MODEL)
    _release_mlx_cache()
    return result.get("text", "").strip()


def _openrouter_transcribe_wav(wav_path: Path) -> list[dict]:
    """Transcribe a file via OpenRouter in chunks of at most _CHUNK_S.

    Returns whisper-style segments with times relative to the whole file.
    """
    audio, rate = _read_wav_mono(wav_path)
    bounds = _chunk_bounds(audio, rate, _CHUNK_S)
    segments = []
    for start, end in zip(bounds, bounds[1:]):
        wav = _wav_bytes(_to_16k(audio[start:end], rate))
        result = _openrouter_request(wav, verbose=True)
        chunk_segments = result.get("segments")
        if not chunk_segments and result.get("text", "").strip():
            # Some providers return text without timestamps; span the chunk.
            chunk_segments = [{"start": 0.0, "end": (end - start) / rate, "text": result["text"]}]
        offset = start / rate
        for seg in chunk_segments or []:
            segments.append({
                "start": seg["start"] + offset,
                "end": seg["end"] + offset,
                "text": seg["text"],
            })
    return segments


def _chunk_bounds(audio: np.ndarray, rate: int, max_s: float) -> list[int]:
    """Sample offsets that split audio into pieces of at most max_s seconds.

    Each cut lands at the quietest 100 ms window in the last 10 s (or last
    half, for short limits) before the limit, so it rarely splits a word.
    """
    max_len = int(max_s * rate)
    win = max(rate // 10, 1)
    search = min(10 * rate, max_len // 2) // win * win
    bounds = [0]
    while len(audio) - bounds[-1] > max_len:
        lo = bounds[-1] + max_len - search
        frames = audio[lo:lo + search].astype(np.float32).reshape(-1, win)
        quietest = int((frames ** 2).mean(axis=1).argmin())
        bounds.append(lo + quietest * win + win // 2)
    if len(audio) > bounds[-1]:
        bounds.append(len(audio))
    return bounds


def _openrouter_request(wav: bytes, verbose: bool) -> dict:
    """POST one 16 kHz mono WAV to OpenRouter's transcription endpoint.

    Retries rate limits (429) and provider-side failures (5xx), honoring
    ``retry-after``. Any other error raises with the response body.
    """
    body = json.dumps({
        "model": _openrouter_model(),
        "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"},
        "response_format": "verbose_json" if verbose else "json",
    }).encode()
    req = urllib.request.Request(_OPENROUTER_URL, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {os.environ['OPENROUTER_API_KEY']}")
    req.add_header("User-Agent", f"hotmic/{__version__}")
    for attempt in range(_OPENROUTER_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=_OPENROUTER_TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:300]
            if (e.code != 429 and e.code < 500) or attempt == _OPENROUTER_RETRIES:
                raise RuntimeError(f"OpenRouter {e.code}: {detail}") from None
            try:
                delay = float(e.headers.get("retry-after", ""))
            except ValueError:
                delay = 2 ** attempt
            time.sleep(min(delay, 60))


def _diarize_segments(wav_path: Path) -> list[tuple] | None:
    """Return [(start, end, speaker), ...], or None if diarization failed.

    Uses the remote service when ``HOTMIC_DIARIZE_URL`` is set (no local
    fallback on failure — returns None so the transcript is written unlabeled).
    Otherwise runs the local ``diarize`` library, for offline single-file use.
    """
    if os.environ.get(_DIARIZE_URL):
        return _diarize_remote(wav_path)
    return _diarize_local(wav_path)


def _diarize_local(wav_path: Path) -> list[tuple] | None:
    try:
        import diarize as diarize_lib
    except ImportError:
        raise SystemExit(
            "diarize is required for local speaker diarization, or set "
            "HOTMIC_DIARIZE_URL to use the remote service.\n"
            "Install it with: pip install -e '.[diarize]'"
        )
    print("Running speaker diarization (local)...")
    result = diarize_lib.diarize(str(wav_path))
    return [(s.start, s.end, s.speaker) for s in result.segments]


def _diarize_remote(wav_path: Path) -> list[tuple] | None:
    """POST 16 kHz mono audio to the remote diarization service."""
    url = os.environ[_DIARIZE_URL].rstrip("/") + "/diarize"
    token = os.environ.get(_DIARIZE_TOKEN, "")
    try:
        audio = _wav_bytes(_to_16k(*_read_wav_mono(wav_path)))
        body, content_type = _multipart_wav(audio, "audio.wav")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", content_type)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        print(f"Diarizing via {os.environ[_DIARIZE_URL]} ...")
        with urllib.request.urlopen(req, timeout=_DIARIZE_TIMEOUT) as resp:
            payload = json.load(resp)
        return [(s["start"], s["end"], s["speaker"]) for s in payload["segments"]]
    except Exception as e:  # network, auth, timeout, malformed response
        print(f"Remote diarization failed ({e}); transcript will be unlabeled.")
        return None


def _read_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read a WAV as int16 mono at its native rate."""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio, rate


def _to_16k(audio: np.ndarray, rate: int) -> np.ndarray:
    """Resample int16 mono audio to 16 kHz (linear interpolation)."""
    if rate == 16000:
        return audio
    n_out = int(len(audio) * 16000 / rate)
    x = np.arange(n_out, dtype=np.float64) * (rate / 16000)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.int16)


def _wav_bytes(audio_16k: np.ndarray) -> bytes:
    """Encode int16 16 kHz mono audio as WAV bytes (small upload)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(audio_16k.tobytes())
    return buf.getvalue()


def _multipart_wav(data: bytes, filename: str) -> tuple[bytes, str]:
    boundary = "----hotmic" + os.urandom(16).hex()
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    return head + data + tail, f"multipart/form-data; boundary={boundary}"


def _find_speaker(speaker_segments: list[tuple], time_point: float) -> str:
    """Find which speaker is talking at a given time point."""
    for start, end, speaker in speaker_segments:
        if start <= time_point <= end:
            return speaker
    # Fallback: find nearest segment
    if not speaker_segments:
        return "Unknown"
    nearest = min(speaker_segments, key=lambda s: min(abs(s[0] - time_point), abs(s[1] - time_point)))
    return nearest[2]


def _format_txt(segments: list[dict]) -> str:
    """Raw transcript only — no speaker labels, even when diarized.

    Speaker attribution is kept in the .srt (which carries timestamps), so the
    .txt stays a clean, readable transcript.
    """
    lines = [seg["text"].strip() for seg in segments if seg["text"].strip()]
    return "\n".join(lines).strip()


def _format_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _srt_timestamp(seg["start"])
        end = _srt_timestamp(seg["end"])
        text = seg["text"].strip()
        if text:
            speaker = seg.get("speaker")
            prefix = f"[{speaker}] " if speaker else ""
            lines.append(f"{i}\n{start} --> {end}\n{prefix}{text}\n")
    return "\n".join(lines)


def _srt_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
