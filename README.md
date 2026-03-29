# hotmic

CLI tool that keeps a rolling mic buffer and saves the last N minutes of audio
on demand.

When something worth keeping happens, hit a hotkey (or CLI commands) and the
last X minutes get written to a WAV file.

Start hotmic listen and it keeps the microphone recording into a
fixed-size rolling buffer. As new audio comes in, the oldest audio gets
discarded, the buffer always holds the most recent N minutes. At any point, you
can save the last X minutes (where X <= N) to a WAV file. Nothing is written to
disk until explicitly asked.

## Install (MacOS)

```bash
brew install portaudio
pip install -e .
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

### [skhd][skhd] integration

```bash
cmd + shift - s : hotmic save 5
cmd + shift - a : hotmic save
cmd + shift - p : hotmic pause
cmd + shift - r : hotmic resume
```

### Options

```
-b --buffer=<min>   Buffer size in minutes [default: 5]
-o --output=<dir>   Output directory [default: ./recordings]
-r --rate=<hz>      Sample rate in Hz [default: 44100]
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
