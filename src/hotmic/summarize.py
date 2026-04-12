"""Summarize transcripts into meeting notes using claude CLI."""

import subprocess
from pathlib import Path

_PROMPT = (
    "You are a meeting notes assistant. Given the following transcript, produce concise "
    "structured notes in Markdown with these sections:\n"
    "## Summary\nOne paragraph overview.\n"
    "## Key Points\nBulleted list of important topics discussed.\n"
    "## Action Items\nBulleted list of follow-ups, owners if mentioned.\n"
    "## Decisions\nBulleted list of decisions made.\n\n"
    "Transcript:\n"
)


def summarize_transcript(txt_path: Path) -> Path:
    """Summarize a transcript file, writing .summary.md alongside it.

    Uses `claude -p` subprocess for generation.
    Returns path to the summary file.
    """
    transcript = txt_path.read_text()
    if not transcript.strip():
        raise ValueError(f"Transcript is empty: {txt_path}")

    prompt = _PROMPT + transcript

    result = subprocess.run(
        ["claude", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        raise RuntimeError(f"claude -p failed: {result.stderr.strip()}")

    summary = result.stdout.strip()
    if not summary:
        raise RuntimeError("claude -p returned empty output")

    summary_path = txt_path.with_suffix(".summary.md")
    summary_path.write_text(summary)
    return summary_path
