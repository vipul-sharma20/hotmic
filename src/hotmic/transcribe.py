"""Transcribe audio files using mlx-whisper, with optional speaker diarization."""

from pathlib import Path

_MODEL = "mlx-community/whisper-large-v3-turbo"


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

    result = mlx_whisper.transcribe(
        str(wav_path),
        path_or_hf_repo=_MODEL,
        word_timestamps=diarize,
    )
    segments = result.get("segments", [])

    if diarize:
        segments = _assign_speakers(wav_path, segments)

    txt_path = wav_path.with_suffix(".txt")
    srt_path = wav_path.with_suffix(".srt")

    txt_path.write_text(_format_txt(segments))
    srt_path.write_text(_format_srt(segments))

    return txt_path, srt_path


def _assign_speakers(wav_path: Path, segments: list[dict]) -> list[dict]:
    """Run diarization and assign speaker labels to whisper segments."""
    try:
        import diarize as diarize_lib
    except ImportError:
        raise SystemExit(
            "diarize is required for speaker diarization.\n"
            "Install it with: pip install -e '.[diarize]'"
        )

    print("Running speaker diarization...")
    result = diarize_lib.diarize(str(wav_path))

    # Build (start, end, speaker) list from diarize output
    speaker_segments = [(s.start, s.end, s.speaker) for s in result.segments]

    # Assign speaker to each whisper segment based on midpoint overlap
    for seg in segments:
        mid = (seg["start"] + seg["end"]) / 2
        speaker = _find_speaker(speaker_segments, mid)
        seg["speaker"] = speaker

    return segments


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
    lines = []
    current_speaker = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        speaker = seg.get("speaker")
        if speaker and speaker != current_speaker:
            current_speaker = speaker
            lines.append(f"\n[{speaker}]")
        lines.append(text)
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
