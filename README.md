# 🎛 Fakers — Beat Mangler

Essentia-powered beat manipulation for audio and video.  
Works locally (macOS / Linux / WSL2) and in Google Colab.

## Requirements

| Dependency | Purpose |
|------------|---------|
| `essentia` | Beat detection (`BeatTrackerMultiFeature`) |
| `pydub` | Audio loading, slicing, stitching, export |
| `numpy` | BPM calculation from inter-beat intervals |
| `tqdm` | Progress bars |
| `moviepy` | Video slicing & export *(optional, video only)* |
| `ffmpeg` | Audio/video decoding & encoding *(system install)* |
