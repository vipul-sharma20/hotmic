"""hotmic: Continuous mic buffer with on-demand save.

Usage:
    hotmic listen [--buffer=<min>] [--output=<dir>] [--rate=<hz>]
    hotmic save [<minutes>]
    hotmic pause
    hotmic resume
    hotmic status
    hotmic -h | --help
    hotmic --version

Options:
    -b --buffer=<min>   Buffer size in minutes [default: 5]
    -o --output=<dir>   Output directory [default: ./recordings]
    -r --rate=<hz>      Sample rate in Hz [default: 44100]
    -h --help           Show this help
    --version           Show version
"""

import atexit
import os
import queue
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path

import sounddevice as sd
from docopt import docopt

from . import __version__
from .ring_buffer import RingBuffer

_PIPE_PATH = "/tmp/hotmic.pipe"


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


def _save(ring: RingBuffer, seconds: int, sample_rate: int, output_dir: Path):
    samples = seconds * sample_rate
    audio = ring.read_last(samples)
    if len(audio) == 0:
        print("Buffer empty, nothing to save.")
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"hotmic_{timestamp}.wav"
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    print(f"Saved {len(audio) / sample_rate:.1f}s -> {filepath}")


def _listen(args):
    buffer_min = float(args["--buffer"])
    sample_rate = int(args["--rate"])
    output_dir = Path(args["--output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    _cleanup_pipe()
    os.mkfifo(_PIPE_PATH)
    atexit.register(_cleanup_pipe)

    capacity = int(buffer_min * 60 * sample_rate)
    ring = RingBuffer(capacity, sample_rate)

    def callback(indata, frames, time_info, status):
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)
        ring.write(indata[:, 0])

    stream = sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
        blocksize=1024,
        callback=callback,
    )

    q = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(q,), daemon=True).start()
    threading.Thread(target=_fifo_reader, args=(q,), daemon=True).start()

    print(f"Listening | buffer: {buffer_min} min | rate: {sample_rate} Hz | output: {output_dir}")
    print("Commands: save [min], pause, resume, status, q")
    print()

    with stream:
        while True:
            cmd = q.get()
            if cmd is None:
                break
            cmd = cmd.lower()

            if not cmd:
                continue
            elif cmd in ("q", "quit", "exit"):
                break
            elif cmd.startswith("save"):
                parts = cmd.split()
                minutes = float(parts[1]) if len(parts) > 1 else buffer_min
                if minutes > buffer_min:
                    print(f"Max buffer is {buffer_min} min, clamping.")
                    minutes = buffer_min
                _save(ring, int(minutes * 60), sample_rate, output_dir)
            elif cmd == "pause":
                if stream.active:
                    stream.stop()
                    print("Paused.")
                else:
                    print("Already paused.")
            elif cmd == "resume":
                if not stream.active:
                    stream.start()
                    print("Resumed.")
                else:
                    print("Already listening.")
            elif cmd == "status":
                filled_s = ring.available / sample_rate
                cap_s = buffer_min * 60
                pct = filled_s / cap_s * 100
                alloc_mb = ring.allocated_bytes / 1_048_576
                max_mb = ring._capacity * 2 / 1_048_576
                state = "listening" if stream.active else "paused"
                print(f"Buffer: {filled_s:.1f}s / {cap_s:.0f}s ({pct:.0f}%) | mem: {alloc_mb:.0f}/{max_mb:.0f} MB | {state}")
            else:
                print(f"Unknown: {cmd}")

    print("Done.")


def main():
    args = docopt(__doc__, version=__version__)

    if args["listen"]:
        _listen(args)
    elif args["save"]:
        minutes = args["<minutes>"] or ""
        _send_command(f"save {minutes}".strip())
    elif args["pause"]:
        _send_command("pause")
    elif args["resume"]:
        _send_command("resume")
    elif args["status"]:
        _send_command("status")
