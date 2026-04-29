# hotmic

CLI tool that keeps a rolling mic buffer and saves the last N minutes of audio
on demand — with transcription, speaker diarization, and AI-powered summaries.

When something worth keeping happens, hit a hotkey (or CLI commands) and the
last X minutes get written to a WAV file. Optionally transcribe it, identify
speakers, and generate meeting notes automatically.

Start hotmic listen and it keeps the microphone recording into a
fixed-size rolling buffer. As new audio comes in, the oldest audio gets
discarded, the buffer always holds the most recent N minutes. At any point, you
can save the last X minutes (where X <= N) to a WAV file. Nothing is written to
disk until explicitly asked.

> [!NOTE]
> This project was built and tested for macOS. It most likely only works on
> macOS, especially system audio capture, which depends on macOS Core Audio
> taps.

## Install (MacOS)

```bash
brew install portaudio
pip install -e .
```

With optional features:

```bash
pip install -e '.[transcribe]'   # + mlx-whisper for transcription
pip install -e '.[diarize]'      # + transcription + speaker diarization
pip install -e '.[all]'          # everything
```

## Usage

```bash
# Start listening with a 30-minute rolling buffer
hotmic listen --buffer 30

# In another terminal (or via hotkey):
hotmic save 5           # save last 5 minutes
hotmic save             # save entire buffer
hotmic pause            # mute mic
hotmic resume           # unmute
hotmic status           # buffer stats (prints in listen terminal)
```

Interactive commands also work directly in the `listen` terminal: `save [min]`, `pause`, `resume`, `status`, `q`.

### System audio capture (meeting recording)

Capture both your mic and system audio (Zoom/Meet/Teams) without touching your audio routing:

```bash
hotmic listen --buffer 60 --system-audio --diarize --summarize
```

Uses [audiotee](https://github.com/makeusabrew/audiotee) to passively tap system audio via macOS Core Audio taps (macOS 14.2+). Your meeting runs normally — no virtual audio drivers, no aggregate devices, no interference.

First run will prompt for "System Audio Recording" permission in System Settings.

### Transcription & summarization

```bash
# Auto-transcribe every save (writes .txt + .srt alongside .wav)
hotmic listen --buffer 30 --transcribe

# Auto-transcribe with speaker diarization
hotmic listen --buffer 30 --diarize

# Auto-transcribe + diarize + generate meeting notes
hotmic listen --buffer 30 --diarize --summarize

# Transcribe an existing file
hotmic transcribe recording.wav
hotmic transcribe recording.wav --diarize

# Summarize an existing transcript
hotmic summarize recording.txt
```

Transcription uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon optimized). Diarization uses [diarize](https://github.com/FoxNoseTech/diarize) (no API keys needed). Summarization uses `claude -p`.

### Bookmarks

Drop timestamp markers during recording, then save specific segments:

```bash
# From another terminal (or via hotkey):
hotmic mark meeting-start    # drop a named bookmark
hotmic mark meeting-end      # drop another
hotmic marks                 # list all marks (in listen terminal)
hotmic save --since-mark     # save from last mark to now
hotmic save --between-marks  # save between last two marks
```

Interactive commands: `mark [label]`, `marks`, `save --since-mark`, `save --between-marks`.

### [skhd][skhd] integration

```bash
cmd + shift - s : hotmic save 5
cmd + shift - a : hotmic save
cmd + shift - m : hotmic mark
cmd + shift - p : hotmic pause
cmd + shift - r : hotmic resume
```

### Options

```
-b --buffer=<min>   Buffer size in minutes [default: 5]
-o --output=<dir>   Output directory [default: ./recordings]
-r --rate=<hz>      Sample rate in Hz [default: 44100]
--system-audio      Capture system audio via audiotee (macOS 14.2+)
--transcribe        Transcribe saved audio using mlx-whisper
--diarize           Identify speakers (requires diarize package)
--summarize         Generate meeting notes (requires claude CLI)
```

Use `--rate 16000` if you only care about voice — cuts memory ~2.75x.

## Memory

RAM grows lazily. Peak when buffer is full:

| Buffer | 44100 Hz | 16000 Hz |
|--------|----------|----------|
| 5 min  | 26 MB    | 9 MB     |
| 30 min | 159 MB   | 58 MB    |
| 60 min | 317 MB   | 115 MB   |


[skhd]: https://github.com/asmvik/skhd
