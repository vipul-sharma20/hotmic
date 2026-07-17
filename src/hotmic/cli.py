"""hotmic: Continuous mic buffer with on-demand save.

Usage:
    hotmic listen [--buffer=<min>] [--output=<dir>] [--rate=<hz>] [--system-audio] [--no-transcribe] [--diarize] [--summarize]
    hotmic save [<minutes>] [--since-mark] [--between-marks] [--name=<name>]
    hotmic pause
    hotmic resume
    hotmic buffer <minutes>
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
    --no-transcribe     Disable transcription (live and on save); on by default
    --diarize           Identify speakers (requires diarize package)
    --summarize         Generate meeting notes after transcription
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

# readline's C init takes the stdin file lock. torch (pulled in by the live
# VAD thread) imports readline transitively, which deadlocks against the
# blocking input() in the stdin reader thread. Importing it here, while only
# the main thread exists, makes the later import a no-op.
try:
    import readline  # noqa: F401
except ImportError:
    pass

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


def _read_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(wav_path), "rb") as wf:
        rate = wf.getframerate()
        channels = wf.getnchannels()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio, rate


def _wav_has_speech(wav_path: Path) -> bool:
    """VAD check on a saved file. Errs toward transcribing if VAD is unavailable."""
    try:
        from .vad import has_speech
        audio, rate = _read_wav_mono(wav_path)
        return has_speech(audio, rate)
    except ImportError:
        return True
    except Exception as e:
        print(f"VAD check failed ({e}), transcribing anyway.", file=sys.stderr)
        return True


def _transcribe_background(wav_path: Path, do_diarize: bool, do_summarize: bool):
    try:
        if not _wav_has_speech(wav_path):
            print(f"No speech detected in {wav_path.parent.name}/{wav_path.name}, skipping transcription.")
            return
        from .transcribe import transcribe_wav
        print(f"Transcribing {wav_path.parent.name}/{wav_path.name}{'  (+ diarization)' if do_diarize else ''}...")
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


# --- Live transcription ---

def _live_transcriber(ring: RingBuffer, sample_rate: int, output_dir: Path,
                      stop: threading.Event, state: dict):
    """Follow the ring buffer, segment speech with Silero VAD, transcribe.

    Polls new audio via the monotonic total_writes cursor. Only audio that
    VAD marks as speech is sent to whisper; silence costs nothing. Each
    finished utterance is appended to a per-session transcript file.
    """
    try:
        from .transcribe import transcribe_audio
        from .vad import SpeechMonitor, to_float_16k
        monitor = SpeechMonitor(sample_rate, 0)
    except ImportError as e:
        print(f"\nLive transcription disabled: {e}", file=sys.stderr)
        return
    except Exception as e:
        print(f"\nLive transcription failed to start: {e}", file=sys.stderr)
        return

    # Audio kept flowing while the VAD model loaded; align coordinates to
    # the ring cursor we will actually start feeding from.
    cursor = ring.total_writes
    monitor.rebase(cursor)

    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcript_path = output_dir / f"live_{session}.txt"
    transcript_path.touch()
    state["transcript"] = transcript_path
    print(f"Live transcript -> {transcript_path}")

    def handle(start_total: int, end_total: int):
        try:
            audio = ring.read_range(start_total, end_total)
        except ValueError:
            return  # utterance already overwritten (buffer smaller than backlog)
        if len(audio) == 0:
            return
        try:
            text = transcribe_audio(to_float_16k(audio, sample_rate))
        except Exception as e:
            print(f"Live transcription error: {e}", file=sys.stderr)
            return
        if not text:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        with open(transcript_path, "a") as f:
            f.write(f"[{ts}] {text}\n")
        state["utterances"] += 1
        print(f"[live {ts}] {text}")

    while not stop.wait(1.0):
        now = ring.total_writes
        if now == cursor:
            continue
        try:
            chunk = ring.read_range(cursor, now)
        except ValueError:
            # Fell behind the buffer (e.g. long transcription) — restart clean.
            monitor = SpeechMonitor(sample_rate, 0)
            cursor = ring.total_writes
            monitor.rebase(cursor)
            continue
        cursor = now
        events = monitor.feed(chunk)
        state["speaking"] = monitor.speaking
        for start_total, end_total in events:
            handle(start_total, end_total)

    for start_total, end_total in monitor.flush():
        handle(start_total, end_total)
    state["speaking"] = False


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


def _parse_buffer_command(cmd: str) -> float:
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"Could not parse buffer command: {e}") from e

    if len(parts) != 2:
        raise ValueError("Usage: buffer <minutes>")

    try:
        minutes = float(parts[1])
    except ValueError as e:
        raise ValueError(f"Invalid minutes value: {parts[1]}") from e

    if minutes <= 0:
        raise ValueError("Buffer minutes must be greater than 0.")

    return minutes


def _parse_transcribe_command(cmd: str) -> tuple[Path, bool]:
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"Could not parse transcribe command: {e}") from e

    file_arg: str | None = None
    do_diarize = False
    for token in parts[1:]:
        if token.lower() == "--diarize":
            do_diarize = True
        elif file_arg is None:
            file_arg = token
        else:
            raise ValueError(f"Unexpected transcribe argument: {token}")

    if file_arg is None:
        raise ValueError("Usage: transcribe <file> [--diarize]")

    return Path(file_arg).expanduser(), do_diarize


def _parse_summarize_command(cmd: str) -> Path:
    try:
        parts = shlex.split(cmd)
    except ValueError as e:
        raise ValueError(f"Could not parse summarize command: {e}") from e

    if len(parts) != 2:
        raise ValueError("Usage: summarize <file>")

    return Path(parts[1]).expanduser()


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


def _stream_watchdog(ring, paused, stop_event, state, restart):
    """Detect a silently-dead mic stream and restart it.

    macOS can kill the AUHAL input stream out from under us on a device
    reconfig (PaMacCore err=-50) — sounddevice stops invoking the callback
    but never raises, so the ring buffer silently freezes. We watch the
    monotonic ``total_writes`` counter; if it stops advancing while we are
    meant to be capturing, we abort and reopen the stream, with backoff so a
    genuinely-gone device (unplugged, no fallback) can't spin.
    """
    POLL = 2.0        # how often to check
    STALL = 5.0       # no new samples for this long => stream is dead
    backoff = 1.0
    last_total = ring.total_writes
    last_advance = time.monotonic()
    while not stop_event.wait(POLL):
        # Paused by the user: mic is intentionally stopped, don't monitor.
        if paused.is_set():
            last_total = ring.total_writes
            last_advance = time.monotonic()
            backoff = 1.0
            state["stalled"] = False
            continue
        now = ring.total_writes
        if now != last_total:
            last_total = now
            last_advance = time.monotonic()
            backoff = 1.0
            state["stalled"] = False
            continue
        if time.monotonic() - last_advance < STALL:
            continue
        # No audio for STALL seconds while unpaused: the stream is dead.
        state["stalled"] = True
        print(f"\n⚠ audio stream stalled (no samples for {STALL:.0f}s) "
              "— restarting", file=sys.stderr)
        if restart():
            state["restarts"] += 1
            state["stalled"] = False
            last_total = ring.total_writes
            last_advance = time.monotonic()
            backoff = 1.0
            print("✓ audio stream restarted", file=sys.stderr)
        else:
            print(f"  restart failed — retrying in {backoff:.0f}s",
                  file=sys.stderr)
            stop_event.wait(backoff)
            backoff = min(backoff * 2, 30.0)
            # keep last_advance stale so we retry on the next poll


# --- Main listen loop ---

def _listen(args):
    buffer_min = float(args["--buffer"])
    sample_rate = int(args["--rate"])
    output_dir = Path(args["--output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    do_system_audio = args.get("--system-audio", False)
    do_transcribe = not args.get("--no-transcribe", False)
    do_diarize = args.get("--diarize", False)
    do_summarize = args.get("--summarize", False)

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

    # The mic stream can die silently on a macOS device reconfig, so it is
    # kept in a mutable holder and reopened by the watchdog on stall. All
    # readers (command loop, watchdog) go through the holder, never a captured
    # `stream` local, so a restart is transparent to them.
    stream_holder = {"stream": None}
    paused = threading.Event()

    def _open_stream():
        s = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            callback=callback,
        )
        s.start()
        return s

    def _restart_stream():
        old = stream_holder["stream"]
        if old is not None:
            old.abort()   # ignore_errors=True by default
            old.close()
        try:
            stream_holder["stream"] = _open_stream()
            return True
        except Exception as e:
            stream_holder["stream"] = None
            print(f"\n[audio] reopen failed: {e}", file=sys.stderr)
            return False

    q = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(q,), daemon=True).start()
    threading.Thread(target=_fifo_reader, args=(q,), daemon=True).start()

    # Live VAD-gated transcription of the ring buffer
    live_stop = threading.Event()
    live_state = {"speaking": False, "utterances": 0, "transcript": None}
    live_thread = None
    if do_transcribe:
        live_thread = threading.Thread(
            target=_live_transcriber,
            args=(ring, sample_rate, output_dir, live_stop, live_state),
        )
        live_thread.start()

    print(f"Listening | buffer: {buffer_min} min | rate: {sample_rate} Hz | output: {output_dir}")
    if do_system_audio:
        print("  System audio: on (via audiotee)")
    print('Commands: save [min] [--name "Meeting"], buffer <min>, mark [label], marks, pause, resume, status, transcribe <file> [--diarize], summarize <file>, q')
    if do_transcribe:
        flags = f"Transcription: on (VAD-gated) | Diarize: {'on' if do_diarize else 'off'} | Auto-summarize: {'on' if do_summarize else 'off'}"
        print(f"  {flags}")
    else:
        print("  Transcription: off")
    print()

    def _launch_post_save(filepath):
        if filepath and do_transcribe:
            t = threading.Thread(
                target=_transcribe_background,
                args=(filepath, do_diarize, do_summarize),
            )
            workers.append(t)
            t.start()

    def _launch_transcribe(wav_path, diarize, summarize):
        if not wav_path.exists():
            print(f"File not found: {wav_path}")
            return
        t = threading.Thread(
            target=_transcribe_background,
            args=(wav_path, diarize, summarize),
        )
        workers.append(t)
        t.start()

    def _launch_summarize(txt_path):
        if not txt_path.exists():
            print(f"File not found: {txt_path}")
            return
        t = threading.Thread(target=_summarize_background, args=(txt_path,))
        workers.append(t)
        t.start()

    stream_holder["stream"] = _open_stream()

    # Watchdog: restarts the mic stream if it dies silently (macOS -50).
    wd_stop = threading.Event()
    wd_state = {"stalled": False, "restarts": 0}
    threading.Thread(
        target=_stream_watchdog,
        args=(ring, paused, wd_stop, wd_state, _restart_stream),
        daemon=True,
    ).start()

    try:
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
            elif command_name in ("buffer", "resize"):
                try:
                    requested_min = _parse_buffer_command(cmd)
                except ValueError as e:
                    print(f"Cannot resize buffer: {e}")
                    continue

                new_capacity = int(requested_min * 60 * sample_rate)
                old_buffer_min = ring.capacity / sample_rate / 60
                if not ring.grow_capacity(new_capacity):
                    if new_capacity == ring.capacity:
                        print(f"Buffer is already {old_buffer_min:g} min.")
                    else:
                        print(
                            f"Current buffer is {old_buffer_min:g} min; "
                            "shrinking is not supported."
                        )
                    continue

                buffer_min = requested_min
                filled_s = ring.available / sample_rate
                alloc_mb = ring.allocated_bytes / 1_048_576
                max_mb = ring.capacity * 2 / 1_048_576
                print(
                    f"Buffer increased: {old_buffer_min:g} -> {buffer_min:g} min "
                    f"| retained: {filled_s:.1f}s | mem: {alloc_mb:.0f}/{max_mb:.0f} MB"
                )
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
                if not paused.is_set():
                    paused.set()   # tell the watchdog to stand down first
                    s = stream_holder["stream"]
                    if s is not None:
                        s.stop()
                    print("Paused.")
                else:
                    print("Already paused.")
            elif command_name == "resume":
                if paused.is_set():
                    s = stream_holder["stream"]
                    if s is not None and not s.active:
                        s.start()
                    elif s is None:
                        stream_holder["stream"] = _open_stream()
                    paused.clear()
                    print("Resumed.")
                else:
                    print("Already listening.")
            elif command_name == "status":
                filled_s = ring.available / sample_rate
                cap_s = ring.capacity / sample_rate
                pct = filled_s / cap_s * 100
                alloc_mb = ring.allocated_bytes / 1_048_576
                max_mb = ring.capacity * 2 / 1_048_576
                if paused.is_set():
                    state = "paused"
                elif wd_state["stalled"]:
                    state = "STALLED (restarting)"
                else:
                    s = stream_holder["stream"]
                    state = "listening" if (s is not None and s.active) else "stopped"
                if wd_state["restarts"]:
                    state += f" | restarts: {wd_state['restarts']}"
                print(f"Buffer: {filled_s:.1f}s / {cap_s:.0f}s ({pct:.0f}%) | mem: {alloc_mb:.0f}/{max_mb:.0f} MB | {state}")
                if live_state["transcript"]:
                    vad = "speaking" if live_state["speaking"] else "quiet"
                    print(f"Live transcript: {live_state['transcript'].name} | VAD: {vad} | {live_state['utterances']} utterance(s)")
                if marks:
                    print(f"Marks: {len(marks)}")
            elif command_name == "transcribe":
                try:
                    wav_path, diarize_arg = _parse_transcribe_command(cmd)
                except ValueError as e:
                    print(f"Cannot transcribe: {e}")
                    continue
                _launch_transcribe(wav_path, diarize_arg, False)
            elif command_name == "summarize":
                try:
                    txt_path = _parse_summarize_command(cmd)
                except ValueError as e:
                    print(f"Cannot summarize: {e}")
                    continue
                _launch_summarize(txt_path)
            else:
                print(f"Unknown: {cmd}")
    finally:
        # Stop the watchdog before closing the stream so it can't reopen it.
        wd_stop.set()
        s = stream_holder["stream"]
        if s is not None:
            s.stop()
            s.close()

    # Stop audiotee if running
    if audiotee_proc:
        audiotee_proc.terminate()
        audiotee_proc.wait(timeout=5)

    # Stop the live transcriber; it flushes any in-progress utterance
    if live_thread:
        live_stop.set()
        live_thread.join(timeout=120)

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
    elif args["buffer"]:
        _send_command(f"buffer {args['<minutes>']}")
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
