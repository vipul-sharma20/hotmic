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
pip install -e '.[transcribe]'   # + mlx-whisper for transcription (not needed with the OpenRouter backend)
pip install -e '.[vad]'          # + silero-vad for live transcription / speech detection
pip install -e '.[diarize]'      # + transcription + speaker diarization
pip install -e '.[all]'          # everything
```

## Usage

```bash
# Start listening with a 30-minute rolling buffer
hotmic listen --buffer 30

# In another terminal (or via hotkey):
hotmic save 5           # save last 5 minutes
hotmic save 5 --name "Weekly Review"
hotmic save             # save entire buffer
hotmic buffer 120       # increase live buffer to 120 minutes
hotmic pause            # mute mic
hotmic resume           # unmute
hotmic status           # buffer stats (prints in listen terminal)
```

Interactive commands also work directly in the `listen` terminal: `save [min] --name "Meeting Name"`, `buffer <min>`, `pause`, `resume`, `status`, `q`.

### System audio capture (meeting recording)

Capture both your mic and system audio (Zoom/Meet/Teams) without touching your audio routing:

```bash
hotmic listen --buffer 60 --system-audio --diarize --summarize
```

Uses [audiotee](https://github.com/makeusabrew/audiotee) to passively tap system audio via macOS Core Audio taps (macOS 14.2+). Your meeting runs normally — no virtual audio drivers, no aggregate devices, no interference.

First run will prompt for "System Audio Recording" permission in System Settings.

When system audio capture is enabled, each save writes:

- `audio.wav` — mixed mono mic + system audio
- `mic.wav` — microphone only
- `system.wav` — system audio only
- `audio_stereo.wav` — stereo split, mic on left and system audio on right
- `metadata.json` — meeting name, save time, duration, sample rate, and file list

### Transcription & summarization

Transcription is **on by default**, gated by [Silero VAD](https://github.com/snakers4/silero-vad):

- **Live transcript**: while listening, speech is detected and segmented into
  utterances, each transcribed as it finishes and appended to a per-session
  `live_<timestamp>.txt` in the output directory. Silence is never sent to
  whisper.
- **On save**: every save is transcribed (`.txt` + `.srt`), unless VAD finds
  no speech in it — then transcription is skipped.

```bash
# Default: live VAD-gated transcription + transcribe every save
hotmic listen --buffer 30

# Disable all transcription
hotmic listen --buffer 30 --no-transcribe

# Auto-transcribe with speaker diarization
hotmic listen --buffer 30 --diarize

# Auto-transcribe + diarize + generate meeting notes
hotmic listen --buffer 30 --diarize --summarize

# Transcribe an existing file
hotmic transcribe recording.wav
hotmic transcribe hotmic_20260429_103000/mic.wav
hotmic transcribe hotmic_20260429_103000/system.wav
hotmic transcribe recording.wav --diarize

# Summarize an existing transcript
hotmic summarize recording.txt
```

Transcription uses [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) (Apple Silicon optimized) by default, or hosted Whisper via OpenRouter (see below). Diarization runs locally via [diarize](https://github.com/FoxNoseTech/diarize), or is offloaded to a remote GPU service (see below). Summarization uses `claude -p`.

### Hosted transcription via OpenRouter

Instead of running whisper on the Mac, hotmic can send speech to
[OpenRouter's speech-to-text endpoint](https://openrouter.ai/docs/guides/overview/multimodal/stt).
The MLX model then never loads (~1.5 GB less RAM) and mlx-whisper is not needed.

```bash
export HOTMIC_TRANSCRIBE_BACKEND=openrouter    # default: mlx
export OPENROUTER_API_KEY=<key>
export HOTMIC_OPENROUTER_MODEL=openai/whisper-large-v3   # optional; default openai/whisper-large-v3-turbo
hotmic listen
```

The backend applies to live transcription, saves, and `hotmic transcribe`.
Having `OPENROUTER_API_KEY` set does not switch backends on its own, so audio
leaves the machine only when you opt in. The startup banner shows the active
backend and model, and `listen` exits early if the config is incomplete.

- Audio is uploaded as 16 kHz mono WAV. Saves longer than 5 minutes are split
  into ≤5 min chunks, cut at the quietest point near each limit, to stay under
  OpenRouter's 25 MB upload cap and 60 s provider timeout.
- Rate limits (429) and provider errors (5xx) are retried up to 3 times,
  honoring `retry-after`.
- Live transcription sends one request per utterance.
- OpenRouter picks the provider for transcription requests and ignores
  `provider.order`/`only`. `openai/whisper-large-v3-turbo` is served by
  DeepInfra and Groq. To route to Groq, add your Groq API key as a
  *prioritized* BYOK key in OpenRouter's integration settings.

### Remote diarization (offload to a GPU box)

Diarization is the most compute-heavy step. Instead of running it on the Mac,
you can offload it to a self-hosted [pyannote](https://github.com/pyannote/pyannote-audio)
service — the **[`diarization-service`](https://github.com/vipul-sharma20/diarization-service)**
project (a separate repo, meant for a GPU machine). Point hotmic at it:

```bash
export HOTMIC_DIARIZE_URL=https://diarize.example.com
export HOTMIC_DIARIZE_TOKEN=<shared secret>
hotmic listen --diarize        # diarization now runs on the remote GPU
```

When set, hotmic downsamples each save to 16 kHz mono and POSTs it to the
service (bearer-token auth), running the request **concurrently** with local
transcription so recording and live transcription are never blocked. If the
service is unreachable the transcript is still written, just without speaker
labels — there is no silent local fallback. The service, its API, and full
deployment/security details live in the separate `diarization-service` project
(`DEPLOY.md` there).

### Bookmarks

Drop timestamp markers during recording, then save specific segments:

```bash
# From another terminal (or via hotkey):
hotmic mark meeting-start    # drop a named bookmark
hotmic mark meeting-end      # drop another
hotmic marks                 # list all marks (in listen terminal)
hotmic save --since-mark --name "Customer Call"     # save from last mark to now
hotmic save --between-marks --name "Design Review"  # save between last two marks
```

Interactive commands: `mark [label]`, `marks`, `save [min] --name "Meeting Name"`, `save --since-mark`, `save --between-marks`.

### Growing the live buffer

Increase retention at runtime without restarting `listen`:

```bash
hotmic buffer 120
```

This preserves audio that is still in the current rolling buffer and allows future audio to fill the larger capacity. Audio already overwritten before the resize cannot be recovered. Shrinking is not supported while recording.

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
--name=<name>       Meeting name to prefix the save directory
--no-transcribe     Disable transcription (live and on save); on by default
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
