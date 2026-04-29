# hotmic

CLI tool that listens on your microphone continuously, maintains a rolling
buffer of the last N minutes (discarding older audio), and lets you save any
portion of it on demand. Supports transcription, speaker diarization, and
AI-powered meeting summaries.

## Project structure

```
src/hotmic/
├── __init__.py        # version
├── __main__.py        # python -m hotmic
├── cli.py             # docopt CLI, sounddevice stream, command loop, save, bookmarks
├── ring_buffer.py     # pre-allocated numpy int16 ring buffer with monotonic write counter
├── transcribe.py      # mlx-whisper transcription + diarize speaker diarization
└── summarize.py       # meeting notes via claude -p
```

## Architecture

- **RingBuffer** (`ring_buffer.py`): Lazily-allocated `np.int16` array with
  write cursor. Starts at ~2 min worth, doubles up to capacity. O(1) writes in
  the audio callback, O(n) reads only on save. Once full, zero allocations.
  Monotonic `_total_writes` counter enables bookmark validity detection.
  `read_range(start_total, end_total)` extracts mixed mono audio between two
  bookmarked positions. `read_last_tracks`/`read_range_tracks` return aligned
  mic and system-audio tracks for split-source output.
- **CLI** (`cli.py`): Subcommand-based. `listen` starts the daemon (docopt,
  sounddevice stream, FIFO + stdin readers, command loop). `save`/`pause`/
  `resume`/`status`/`mark`/`marks` send commands to the running daemon via a
  named pipe (`/tmp/hotmic.pipe`). Transcription and summarization run on
  non-daemon background threads that complete even if the user quits.
- **Transcribe** (`transcribe.py`): mlx-whisper for transcription (Apple Silicon
  optimized), `diarize` library for speaker identification. Writes `.txt` and
  `.srt` files alongside the WAV.
- **Summarize** (`summarize.py`): Sends transcript to `claude -p` subprocess,
  writes `.summary.md` with structured meeting notes.

## Key design decisions

- **int16 capture**: sounddevice captures directly as int16. No float32→int16 conversion. Halves memory vs float32.
- **Lazy growth**: Starts at ~10 MB (2 min at 44100 Hz), doubles as buffer fills. No wasted RAM if you quit early. Once at capacity, never allocates again.
- **Single numpy array**: Avoids millions of small array allocations that a deque-of-chunks approach would create.
- **Configurable sample rate**: Default 44100 Hz. Use `--rate 16000` for voice-only (cuts memory ~2.75x).
- **Split source saves**: With `--system-audio`, each save keeps compatible
  mixed mono output in `audio.wav` and also writes `mic.wav`, `system.wav`, and
  `audio_stereo.wav` with mic on the left channel and system audio on the right.
- **Meeting names**: `hotmic save --name "Meeting Name"` prefixes the
  timestamped save directory with a sanitized name and writes the original name to
  `metadata.json`.
- **Monotonic write counter for bookmarks**: `_pos` wraps around; `_total_writes` only increments. Bookmark validity is checked against current `_total_writes` to detect overwritten audio.
- **Non-daemon background threads**: Transcription/summarization threads are non-daemon and joined on exit, so they complete even if the user quits right after saving.
- **Optional deps with lazy imports**: Base install is numpy+sounddevice+docopt. mlx-whisper and diarize are optional.
- **`diarize` over pyannote**: No HuggingFace token needed, ~7x faster on CPU, Apache 2.0 license.
- **`claude -p` for summarization**: No API key management needed, uses the user's existing Claude CLI auth.

## Memory footprint

Peak (when buffer is full):

| Buffer | 44100 Hz | 16000 Hz |
|--------|----------|----------|
| 5 min  | 26 MB    | 9 MB     |
| 30 min | 159 MB   | 58 MB    |
| 60 min | 317 MB   | 115 MB   |

Actual RAM grows lazily — starts at ~10 MB, doubles as needed. Use `status` command to see current allocation.

## Install & run

```bash
# dev install
pip install -e .

# with transcription + diarization
pip install -e '.[all]'

# or run directly
python -m hotmic listen --buffer 10 --output ./recordings

# after pip install
hotmic listen --buffer 10
```

Commands while running: `save [min]`, `mark [label]`, `marks`, `pause`, `resume`, `status`, `q`

External commands (from another terminal or skhd):
```bash
hotmic save 5              # save last 5 minutes
hotmic save                # save entire buffer
hotmic save --since-mark   # save from last mark to now
hotmic save --between-marks # save between last two marks
hotmic mark meeting-start  # drop a bookmark
hotmic marks               # list bookmarks
hotmic pause               # stop mic
hotmic resume              # restart mic
hotmic status              # print buffer info (in listen terminal)
hotmic transcribe file.wav --diarize  # transcribe + identify speakers
hotmic summarize file.txt  # generate meeting notes
```

## Dependencies

Core:
- `numpy` — ring buffer backing array + audio data
- `sounddevice` — PortAudio bindings (needs system lib: `brew install portaudio`)
- `docopt-ng` — CLI argument parsing from usage docstring

Optional:
- `mlx-whisper` — transcription (Apple Silicon optimized Whisper)
- `diarize` — speaker diarization (Silero VAD + WeSpeaker + spectral clustering)
- `claude` CLI — summarization (must be installed separately)
