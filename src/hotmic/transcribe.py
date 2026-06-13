"""Transcribe audio files using mlx-whisper, with optional speaker diarization.

Diarization is offloaded to a remote pyannote service when
``HOTMIC_DIARIZE_URL`` is set. The request runs concurrently with local
MLX transcription and is merged at the end, so it never sits on the recording
or live-transcription path. If the service is unreachable the transcript is
still written, just without speaker labels — there is no local fallback once a
remote backend is configured.
"""

import io
import json
import os
import threading
import urllib.request
import wave
from pathlib import Path

import numpy as np

_MODEL = "mlx-community/whisper-large-v3-turbo"

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


def transcribe_wav(wav_path: Path, diarize: bool = False) -> tuple[Path, Path]:
    """Transcribe a WAV file, writing .txt and .srt alongside it.

    If diarize=True, runs speaker diarization and labels each
    segment with a speaker ID.

    Returns (txt_path, srt_path).
    """
    try:
        import mlx_whisper
    except ImportError:
        raise SystemExit(
            "mlx-whisper is required for transcription.\n"
            "Install it with: pip install -e '.[transcribe]'"
        )

    # Kick off diarization (remote GPU or local lib) before transcribing, so
    # it overlaps the local MLX work instead of running after it.
    diar = {}
    diar_thread = None
    if diarize:
        diar_thread = threading.Thread(
            target=lambda: diar.update(segments=_diarize_segments(wav_path)),
            daemon=True,
        )
        diar_thread.start()

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
        audio = _wav_to_16k_mono_wav_bytes(wav_path)
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


def _wav_to_16k_mono_wav_bytes(wav_path: Path) -> bytes:
    """Read a WAV and return 16 kHz mono int16 WAV bytes (small upload)."""
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != 16000:
        n_out = int(len(audio) * 16000 / rate)
        x = np.arange(n_out, dtype=np.float64) * (rate / 16000)
        audio = np.interp(x, np.arange(len(audio)), audio).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(audio.tobytes())
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
