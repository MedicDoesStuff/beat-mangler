# 🎛 Fakers — Beat Mangler

Essentia-powered beat manipulation for audio and video.  
Works locally (macOS / Linux / WSL2) and in Google Colab.

| Mode | What it does |
|------|--------------|
| `remove` | Drops every other beat — half-time feel |
| `swap` | Swaps beats 2 & 4 in every bar of 4 |
| `reverse` | Reverses the entire beat order |
| `shuffle` | Randomises all beat positions |
| `repeat` | Stutters — each beat plays N times in a row |
| `interleave` | Alternates beats from two different files |

---

## Repository structure

```
beat_mangler/
├── beat_mangler.py          # Core library — importable from notebooks or scripts
├── cli.py             # Interactive command-line interface
├── Fakers_Colab.ipynb # Notebook for Google Colab (or local Jupyter)
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup — Google Colab

1. Open `Fakers_Colab.ipynb` in Colab  
   *(File → Open notebook → GitHub tab → paste your repo URL)*

2. In **Cell 1**, replace the placeholder with your actual repo URL:
   ```python
   REPO_URL = "https://github.com/YOUR_USERNAME/beat_mangler.git"
   ```

3. Run all cells top to bottom.  
   Cell 1 installs dependencies and clones the repo.  
   Cell 2 lets you upload your audio/video file(s).  
   Cell 3 is where you pick your mode and run the processing.

---

## Setup — Local (macOS / Linux / WSL2)

### 1 · Install ffmpeg

**macOS (Homebrew)**
```bash
brew install ffmpeg
```

**Ubuntu / Debian / WSL2**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

### 2 · Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/beat_mangler.git
cd beat_mangler
```

### 3 · Create a virtual environment and install Python deps

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows WSL2: same command
pip install -r requirements.txt
```

> **Note:** `essentia` publishes pre-built wheels for Linux and macOS.  
> Native Windows is not supported — use WSL2 instead.

### 4 · Run the CLI

```bash
python cli.py
```

You will be prompted to choose a mode and enter file paths interactively.

### 5 · Or use the library directly in your own script

```python
import beat_mangler

# Single-file modes
out = beat_mangler.process_audio("song.mp3", mode="remove", fmt="flac")
out = beat_mangler.process_audio("song.flac", mode="shuffle", fmt="mp3")
out = beat_mangler.process_audio("song.mp3", mode="repeat", fmt="flac", repeat_times=3)
out = beat_mangler.process_video("clip.mp4", mode="reverse")

# Interleave two tracks
out = beat_mangler.process_interleave(
    path_a="track_a.mp3",
    path_b="track_b.mp3",
    group=2,       # 2 beats from A, then 2 from B, repeat
    fmt="flac",
)
```

---

## How to create the GitHub repository

1. Go to [github.com/new](https://github.com/new) and create a new **public** (or private) repository named `beat_mangler`.

2. Push the files:
   ```bash
   cd beat_mangler
   git init
   git add .
   git commit -m "initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/beat_mangler.git
   git push -u origin main
   ```

3. Done. Open `Fakers_Colab.ipynb` in Colab via the GitHub tab and update `REPO_URL`.

---

## Requirements

| Dependency | Purpose |
|------------|---------|
| `essentia` | Beat detection (`BeatTrackerMultiFeature`) |
| `pydub` | Audio loading, slicing, stitching, export |
| `numpy` | BPM calculation from inter-beat intervals |
| `tqdm` | Progress bars |
| `moviepy` | Video slicing & export *(optional, video only)* |
| `ffmpeg` | Audio/video decoding & encoding *(system install)* |
