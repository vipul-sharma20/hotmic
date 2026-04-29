"""hotmic: Continuous mic buffer with on-demand save.

Usage:
    hotmic listen [--buffer=<min>] [--output=<dir>] [--rate=<hz>] [--system-audio] [--transcribe] [--diarize] [--summarize]
    hotmic save [<minutes>] [--since-mark] [--between-marks] [--name=<name>]
    hotmic pause
    hotmic resume
    hotmic status
    hotmic mark [<label>]
    hotmic marks
    hotmic transcribe <file> [--diarize]
    hotmic summarize <file>
    hotmic -h | --help
    hotmic --version

Options:
    -b --buffer=<min>   Buffer size in minutes [default: 5]
    -o --output=<dir>   Output directory [default: ./recordings]
    -r --rate=<hz>      Sample rate in Hz [default: 44100]
    --system-audio      Capture system audio (Zoom/Meet/Teams) via audiotee
    --name=<name>       Meeting name to prefix the save directory
    --transcribe        Transcribe saved audio using mlx-whisper
    --diarize           Identify speakers (requires diarize package)
    --summarize         Generate meeting notes after transcription (implies --transcribe)
    -h --help           Show this help
    --version           Show version
"""

import atexit
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
from docopt import docopt

from . import __version__
from .ring_buffer import RingBuffer

_PIPE_PATH = "/tmp/hotmic.pipe"
_AUDIOTEE_BIN = Path(__file__).parent.parent.parent / "bin" / "audiotee"


def _send_command(cmd: str):
    """Send a command to the running listen process via FIFO."""
    if not os.path.exists(_PIPE_PATH):
        print("hotmic is not running. Start it with: hotmic listen", file=sys.stderr)
        sys.exit(1)
    with open(_PIPE_PATH, "w") as f:
        f.write(cmd + "\n")


def _cleanup_pipe():
    if os.path.exists(_PIPE_PATH):
        os.unlink(_PIPE_PATH)


def _stdin_reader(q: queue.Queue):
    try:
        while True:
            line = input("> ")
            q.put(line.strip())
    except (KeyboardInterrupt, EOFError):
        q.put(None)


def _fifo_reader(q: queue.Queue):
    while True:
        try:
            with open(_PIPE_PATH) as f:
                for line in f:
                    cmd = line.strip()
                    if cmd:
                        q.put(cmd)
        except OSError:
            break


def _write_wav(audio, sample_rate: int, filepath: Path):
    channels = 1 if audio.ndim == 1 else audio.shape[1]
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())


def _slugify_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug[:80]


def _create_save_dir(output_dir: Path, meeting_name: str | None = None) -> Path:
    """Create a timestamped directory for this save's outputs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify_name(meeting_name) if meeting_name else ""
    dirname = f"{slug}_hotmic_{timestamp}" if slug else f"hotmic_{timestamp}"
    save_dir = output_dir / dirname
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _write_recording_files(
    save_dir: Path,
    primary: np.ndarray,
    aux: np.ndarray,
    sample_rate: int,
    split_sources: bool,
) -> Path:
    mixed = RingBuffer.mix_tracks(primary, aux)
    filepath = save_dir / "audio.wav"
    _write_wav(mixed, sample_rate, filepath)

    if split_sources:
        _write_wav(primary, sample_rate, save_dir / "mic.wav")
        _write_wav(aux, sample_rate, save_dir / "system.wav")
        stereo = np.column_stack((primary, aux)).astype(np.int16, copy=False)
        _write_wav(stereo, sample_rate, save_dir / "audio_stereo.wav")

    return filepath


def _write_save_metadata(
    save_dir: Path,
    meeting_name: str | None,
    duration_seconds: float,
    sample_rate: int,
    split_sources: bool,
):
    files = ["audio.wav"]
    if split_sources:
        files.extend(["mic.wav", "system.wav", "audio_stereo.wav"])
    metadata = {
        "meeting_name": meeting_name or "",
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "files": files,
    }
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _save(
    ring: RingBuffer,
    seconds: int,
    sample_rate: int,
    output_dir: Path,
    split_sources: bool = False,
    meeting_name: str | None = None,
):
    samples = seconds * sample_rate
    primary, aux = ring.read_last_tracks(samples)
    if len(primary) == 0:
        print("Buffer empty, nothing to save.")
        return None
    save_dir = _create_save_dir(output_dir, meeting_name)
    filepath = _write_recording_files(save_dir, primary, aux, sample_rate, split_sources)
    _write_save_metadata(save_dir, meeting_name, len(primary) / sample_rate,
                         sample_rate, split_sources)
    extras = " (+ mic.wav, system.wav, audio_stereo.wav)" if split_sources else ""
    print(f"Saved {len(primary) / sample_rate:.1f}s -> {save_dir.name}/{extras}")
    return filepath


def _save_range(ring: RingBuffer, start_total: int, end_total: int,
                sample_rate: int, output_dir: Path, split_sources: bool = False,
                meeting_name: str | None = None):
    try:
        primary, aux = ring.read_range_tracks(start_total, end_total)
    except ValueError as e:
        print(f"Cannot save: {e}")
        return None
    if len(primary) == 0:
        print("No audio in range.")
        return None
    save_dir = _create_save_dir(output_dir, meeting_name)
    filepath = _write_recording_files(save_dir, primary, aux, sample_rate, split_sources)
    _write_save_metadata(save_dir, meeting_name, len(primary) / sample_rate,
                         sample_rate, split_sources)
    extras = " (+ mic.wav, system.wav, audio_stereo.wav)" if split_sources else ""
    print(f"Saved {len(primary) / sample_rate:.1f}s -> {save_dir.name}/{extras}")
    return filepath


def _transcribe_background(wav_path: Path, do_diarize: bool, do_summarize: bool):
    try:
        from .transcribe import transcribe_wav
        print(f"Transcribing {wav_path.parent.name}/audio.wav{'  (+ diarization)' if do_diarize else ''}...")
        txt_path, srt_path = transcribe_wav(wav_path, diarize=do_diarize)
        print(f"Transcribed -> {wav_path.parent.name}/{txt_path.name}, {srt_path.name}")
        if do_summarize:
            _summarize_background(txt_path)
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)


def _summarize_background(txt_path: Path):
    try:
        from .summarize import summarize_transcript
        print(f"Summarizing {txt_path.parent.name}/{txt_path.name}...")
        summary_path = summarize_transcript(txt_path)
        print(f"Summary -> {txt_path.parent.name}/{summary_path.name}")
    except Exception as e:
        print(f"Summarization failed: {e}", file=sys.stderr)


# --- Marks persistence ---

def _marks_file(output_dir: Path) -> Path:
    return output_dir / "marks.json"


def _load_marks(output_dir: Path) -> list[dict]:
    path = _marks_file(output_dir)
    if path.exists():
        return json.loads(path.read_text())
    return []


def _save_marks(output_dir: Path, marks: list[dict]):
    path = _marks_file(output_dir)
    path.write_text(json.dumps(marks, indent=2))


def _append_mark(output_dir: Path, marks: list[dict], total_writes: int, wall: float, label: str):
    mark = {"total_writes": total_writes, "time": wall, "label": label}
    marks.append(mark)
    _save_marks(output_dir, marks)


# --- audiotee ---

def _parse_save_command(cmd: str) -> tuple[set[str], float | None, str | None]:
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"Could not parse save command: {e}") from e

    flags: set[str] = set()
    minutes: float | None = None
    meeting_name: str | None = None
    i = 1
    while i < len(parts):
        token = parts[i]
        lower = token.lower()
        if lower in ("--between-marks", "--since-mark"):
            flags.add(lower)
        elif lower.startswith("--name="):
            meeting_name = token.split("=", 1)[1].strip()
        elif lower == "--name":
            i += 1
            if i >= len(parts):
                raise ValueError("--name needs a meeting name.")
            meeting_name = parts[i].strip()
        elif minutes is None:
            try:
                minutes = float(token)
            except ValueError as e:
                raise ValueError(f"Invalid minutes value: {token}") from e
        else:
            raise ValueError(f"Unexpected save argument: {token}")
        i += 1

    return flags, minutes, meeting_name or None


def _find_audiotee() -> Path:
    """Find the audiotee binary."""
    if _AUDIOTEE_BIN.exists():
        return _AUDIOTEE_BIN
    found = shutil.which("audiotee")
    if found:
        return Path(found)
    raise FileNotFoundError(
        "audiotee binary not found. Build it from https://github.com/makeusabrew/audiotee\n"
        "and place it in bin/audiotee or add it to your PATH."
    )


def _audiotee_reader(ring: RingBuffer, sample_rate: int, proc: subprocess.Popen):
    """Read PCM int16 chunks from audiotee stdout and write to aux ring buffer."""
    chunk_samples = 1024
    chunk_bytes = chunk_samples * 2  # int16 = 2 bytes per sample
    try:
        while True:
            data = proc.stdout.read(chunk_bytes)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16)
            ring.write_aux(samples)
    except Exception as e:
        print(f"\n[system-audio] {e}", file=sys.stderr)


# --- Main listen loop ---

def _listen(args):
    buffer_min = float(args["--buffer"])
    sample_rate = int(args["--rate"])
    output_dir = Path(args["--output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    do_system_audio = args.get("--system-audio", False)
    do_transcribe = args.get("--transcribe", False)
    do_diarize = args.get("--diarize", False)
    do_summarize = args.get("--summarize", False)
    if do_summarize:
        do_transcribe = True
    if do_diarize:
        do_transcribe = True

    _cleanup_pipe()
    os.mkfifo(_PIPE_PATH)
    atexit.register(_cleanup_pipe)

    capacity = int(buffer_min * 60 * sample_rate)
    ring = RingBuffer(capacity, sample_rate)

    # Bookmarks — persisted to marks.json
    marks: list[dict] = _load_marks(output_dir)
    if marks:
        print(f"Loaded {len(marks)} mark(s) from previous session.")

    # Background workers (non-daemon) so they finish on exit
    workers: list[threading.Thread] = []

    # System audio capture via audiotee
    audiotee_proc = None
    if do_system_audio:
        audiotee_bin = _find_audiotee()
        audiotee_proc = subprocess.Popen(
            [str(audiotee_bin), "--sample-rate", str(sample_rate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(
            target=_audiotee_reader,
            args=(ring, sample_rate, audiotee_proc),
            daemon=True,
        ).start()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        ring.write(indata[:, 0])

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        callback=callback,
    )

    q = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(q,), daemon=True).start()
    threading.Thread(target=_fifo_reader, args=(q,), daemon=True).start()

    print(f"Listening | buffer: {buffer_min} min | rate: {sample_rate} Hz | output: {output_dir}")
    if do_system_audio:
        print("  System audio: on (via audiotee)")
    print('Commands: save [min] [--name "Meeting"], mark [label], marks, pause, resume, status, q')
    if do_transcribe:
        flags = f"Auto-transcribe: on | Diarize: {'on' if do_diarize else 'off'} | Auto-summarize: {'on' if do_summarize else 'off'}"
        print(f"  {flags}")
    print()

    def _launch_post_save(filepath):
        if filepath and do_transcribe:
            t = threading.Thread(
                target=_transcribe_background,
                args=(filepath, do_diarize, do_summarize),
            )
            workers.append(t)
            t.start()

    with stream:
        while True:
            cmd = q.get()
            if cmd is None:
                break
            command_name = cmd.split(maxsplit=1)[0].lower() if cmd else ""

            if not cmd:
                continue
            elif command_name in ("q", "quit", "exit"):
                break
            elif command_name == "save":
                try:
                    flags, minutes_arg, meeting_name = _parse_save_command(cmd)
                except ValueError as e:
                    print(f"Cannot save: {e}")
                    continue

                if "--between-marks" in flags:
                    if len(marks) < 2:
                        print("Need at least 2 marks for --between-marks.")
                    else:
                        start_total = marks[-2]["total_writes"]
                        end_total = marks[-1]["total_writes"]
                        filepath = _save_range(ring, start_total, end_total,
                                               sample_rate, output_dir, do_system_audio,
                                               meeting_name)
                        _launch_post_save(filepath)
                elif "--since-mark" in flags:
                    if not marks:
                        print("No marks set. Use 'mark' first.")
                    else:
                        start_total = marks[-1]["total_writes"]
                        end_total = ring.total_writes
                        filepath = _save_range(ring, start_total, end_total,
                                               sample_rate, output_dir, do_system_audio,
                                               meeting_name)
                        _launch_post_save(filepath)
                else:
                    minutes = minutes_arg if minutes_arg is not None else buffer_min
                    if minutes > buffer_min:
                        print(f"Max buffer is {buffer_min} min, clamping.")
                        minutes = buffer_min
                    filepath = _save(ring, int(minutes * 60), sample_rate,
                                     output_dir, do_system_audio, meeting_name)
                    _launch_post_save(filepath)
            elif command_name == "mark":
                parts = cmd.split(maxsplit=1)
                label = parts[1] if len(parts) > 1 else ""
                tw = ring.total_writes
                wall = time.time()
                _append_mark(output_dir, marks, tw, wall, label)
                ts = datetime.fromtimestamp(wall).strftime("%H:%M:%S")
                idx = len(marks) - 1
                name = f" '{label}'" if label else ""
                print(f"Mark #{idx}{name} at {ts}")
            elif command_name == "marks":
                if not marks:
                    print("No marks.")
                else:
                    now_tw = ring.total_writes
                    oldest = now_tw - ring.available
                    for i, m in enumerate(marks):
                        ts = datetime.fromtimestamp(m["time"]).strftime("%H:%M:%S")
                        valid = "ok" if m["total_writes"] >= oldest else "overwritten"
                        name = f" '{m['label']}'" if m["label"] else ""
                        print(f"  #{i}{name} at {ts} [{valid}]")
            elif command_name == "pause":
                if stream.active:
                    stream.stop()
                    print("Paused.")
                else:
                    print("Already paused.")
            elif command_name == "resume":
                if not stream.active:
                    stream.start()
                    print("Resumed.")
                else:
                    print("Already listening.")
            elif command_name == "status":
                filled_s = ring.available / sample_rate
                cap_s = buffer_min * 60
                pct = filled_s / cap_s * 100
                alloc_mb = ring.allocated_bytes / 1_048_576
                max_mb = ring._capacity * 2 / 1_048_576
                state = "listening" if stream.active else "paused"
                print(f"Buffer: {filled_s:.1f}s / {cap_s:.0f}s ({pct:.0f}%) | mem: {alloc_mb:.0f}/{max_mb:.0f} MB | {state}")
                if marks:
                    print(f"Marks: {len(marks)}")
            else:
                print(f"Unknown: {cmd}")

    # Stop audiotee if running
    if audiotee_proc:
        audiotee_proc.terminate()
        audiotee_proc.wait(timeout=5)

    # Wait for background transcription/summarization to finish
    alive = [t for t in workers if t.is_alive()]
    if alive:
        print(f"Waiting for {len(alive)} background task(s)...")
        for t in alive:
            t.join(timeout=300)

    print("Done.")


def main():
    args = docopt(__doc__, version=__version__)

    if args["listen"]:
        _listen(args)
    elif args["save"]:
        parts = []
        if args["<minutes>"]:
            parts.append(args["<minutes>"])
        if args["--since-mark"]:
            parts.append("--since-mark")
        if args["--between-marks"]:
            parts.append("--between-marks")
        if args["--name"]:
            parts.extend(["--name", shlex.quote(args["--name"])])
        _send_command(f"save {' '.join(parts)}".strip())
    elif args["pause"]:
        _send_command("pause")
    elif args["resume"]:
        _send_command("resume")
    elif args["status"]:
        _send_command("status")
    elif args["mark"]:
        label = args["<label>"] or ""
        _send_command(f"mark {label}".strip())
    elif args["marks"]:
        _send_command("marks")
    elif args["transcribe"]:
        from .transcribe import transcribe_wav
        wav_path = Path(args["<file>"])
        if not wav_path.exists():
            print(f"File not found: {wav_path}", file=sys.stderr)
            sys.exit(1)
        do_diarize = args.get("--diarize", False)
        print(f"Transcribing {wav_path.name}{'  (+ diarization)' if do_diarize else ''}...")
        txt_path, srt_path = transcribe_wav(wav_path, diarize=do_diarize)
        print(f"Done -> {txt_path}, {srt_path}")
    elif args["summarize"]:
        from .summarize import summarize_transcript
        txt_path = Path(args["<file>"])
        if not txt_path.exists():
            print(f"File not found: {txt_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Summarizing {txt_path.name}...")
        summary_path = summarize_transcript(txt_path)
        print(f"Done -> {summary_path}")
