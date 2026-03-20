"""
fakers.py — Beat Mangler core library
Essentia-powered beat manipulation for audio & video.
Works locally and in Google Colab.
"""

import base64
import json
import os
import random
import re
import subprocess
import tempfile
import threading
import time

import essentia.standard as es
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from tqdm.auto import tqdm

# Optional heavy imports (only needed for video)
try:
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    _MOVIEPY_AVAILABLE = True
except ImportError:
    _MOVIEPY_AVAILABLE = False


# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}

AUDIO_FMT_MAP = {
    "mp3": "mp3", "wav": "wav", "flac": "flac",
    "ogg": "ogg", "aac": "aac", "m4a": "mp4", "aiff": "aiff",
}

MODE_SUFFIXES = {
    "remove":     "every_other_removed",
    "swap":       "beats_swapped",
    "reverse":    "beats_reversed",
    "shuffle":    "beats_shuffled",
    "repeat":     "beats_repeated",
    "interleave": "interleaved",
}

SPINNER    = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
SIMPLE_BAR = "{l_bar}{bar}| {elapsed}"


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def src_fmt(path: str) -> str:
    return AUDIO_FMT_MAP.get(os.path.splitext(path)[1].lower().lstrip("."), "mp3")


def make_output_path(input_path: str, suffix: str, ext: str) -> str:
    return f"{os.path.splitext(input_path)[0]}_{suffix}.{ext}"


def simple_bar(desc: str):
    return tqdm(total=1, desc=desc, bar_format=SIMPLE_BAR)


def load_audio(path: str) -> AudioSegment:
    return AudioSegment.from_file(path, format=src_fmt(path))


# ════════════════════════════════════════════════════════════════════════════
# VIDEO PROBING
# ════════════════════════════════════════════════════════════════════════════

def probe_video(path: str) -> dict:
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", "-show_format", path]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)

    vs  = next((s for s in info.get("streams", []) if s["codec_type"] == "video"), {})
    as_ = next((s for s in info.get("streams", []) if s["codec_type"] == "audio"), {})

    num, den = vs.get("r_frame_rate", "30/1").split("/")
    raw_br   = vs.get("bit_rate") or info.get("format", {}).get("bit_rate", "0")

    return {
        "width":        int(vs.get("width",  1920)),
        "height":       int(vs.get("height", 1080)),
        "fps":          round(int(num) / int(den), 3),
        "bitrate_kbps": max(1000, int(raw_br) // 1000),
        "pix_fmt":      vs.get("pix_fmt", "yuv420p"),
        "audio_sr":     int(as_.get("sample_rate", 44100)),
        "audio_ch":     int(as_.get("channels", 2)),
        "audio_kbps":   max(128, int(as_.get("bit_rate", "192000")) // 1000),
    }


def build_video_export_params(probe: dict) -> dict:
    br  = probe["bitrate_kbps"]
    crf = "16" if br >= 8000 else "18" if br >= 4000 else "20" if br >= 2000 else "22" if br >= 1000 else "24"
    audio_kbps = min(320, max(128, probe["audio_kbps"]))

    print(f"\n   📐 {probe['width']}×{probe['height']}  {probe['fps']} fps")
    print(f"   🎞  Source bitrate ~{br} kbps → CRF {crf}")
    print(f"   🔊 Audio {audio_kbps} kbps  ·  {probe['audio_sr']} Hz  ·  {probe['audio_ch']}ch")

    return dict(
        codec="libx264", audio_codec="aac",
        fps=probe["fps"], bitrate=f"{br}k",
        audio_bitrate=f"{audio_kbps}k", audio_fps=probe["audio_sr"],
        preset="slow", verbose=False,
        ffmpeg_params=["-crf", crf, "-pix_fmt", probe["pix_fmt"]],
    )


# ════════════════════════════════════════════════════════════════════════════
# BEAT DETECTION
# ════════════════════════════════════════════════════════════════════════════

def extract_wav(path: str) -> tuple:
    """Return a mono 44 100 Hz WAV temp file and source duration in seconds."""
    tmp = tempfile.mktemp(suffix=".wav")
    label = "📤 Extracting audio" if is_video(path) else "📤 Converting audio"
    with simple_bar(label) as pb:
        if is_video(path):
            if not _MOVIEPY_AVAILABLE:
                raise ImportError("moviepy is required for video. Install it with: pip install moviepy")
            clip = VideoFileClip(path)
            clip.audio.write_audiofile(
                tmp, fps=44100, nbytes=2,
                codec="pcm_s16le", verbose=False, logger=None,
            )
            dur = clip.duration
            clip.close()
        else:
            AudioSegment.from_file(path, format=src_fmt(path)) \
                .set_channels(1).set_frame_rate(44100).export(tmp, format="wav")
            dur = AudioSegment.from_file(path).duration_seconds
        pb.update(1)
    return tmp, dur


def extract_wav_from_segment(seg: AudioSegment) -> str:
    """Export an in-memory AudioSegment to a temp mono WAV for Essentia."""
    tmp = tempfile.mktemp(suffix=".wav")
    seg.set_channels(1).set_frame_rate(44100).export(tmp, format="wav")
    return tmp


def detect_beats(wav_path: str, label: str = "") -> tuple:
    """Run Essentia BeatTrackerMultiFeature in a thread; show a spinner."""
    result = {}

    def _run():
        audio = es.MonoLoader(filename=wav_path, sampleRate=44100)()
        bt, conf = es.BeatTrackerMultiFeature()(audio)
        bpm = 60.0 / np.median(np.diff(bt)) if len(bt) > 1 else 0.0
        result.update({"bt": list(bt), "bpm": bpm,
                        "dur": len(audio) / 44100, "conf": conf})

    t = threading.Thread(target=_run)
    t.start()
    for i in range(10_000):
        if not t.is_alive():
            break
        tag = f" {label}" if label else ""
        print(f"\r🎵 Detecting beats{tag} {SPINNER[i % len(SPINNER)]} ",
              end="", flush=True)
        time.sleep(0.1)
    t.join()
    tag = f" {label}" if label else ""
    print(f"\r🎵 Beats detected!{tag}                    ")

    bt = result["bt"]
    print(f"   {len(bt)} beats  ·  {result['bpm']:.1f} BPM  ·  confidence {result['conf']:.3f}")
    return bt, result["bpm"], result["dur"]


# ════════════════════════════════════════════════════════════════════════════
# SILENCE STRIPPING & BEAT ALIGNMENT  (used by interleave mode)
# ════════════════════════════════════════════════════════════════════════════

def strip_silence(audio: AudioSegment, label: str = "",
                  thresh_db: float = -50, chunk_ms: int = 10):
    """Remove leading & trailing silence before beat detection."""
    lead  = detect_leading_silence(audio, silence_threshold=thresh_db, chunk_size=chunk_ms)
    trail = detect_leading_silence(audio.reverse(), silence_threshold=thresh_db, chunk_size=chunk_ms)
    total = len(audio)
    lead  = min(lead,  total)
    trail = min(trail, total - lead)
    trimmed = audio[lead : total - trail]
    if lead > 0 or trail > 0:
        print(f"   🔇 {label}: stripped {lead}ms leading + {trail}ms trailing silence")
    else:
        print(f"   ✓  {label}: no edge silence detected")
    return trimmed, lead / 1000.0


def align_to_first_beat(beat_times: list, audio: AudioSegment, label: str = ""):
    """Snap audio start to the first detected beat and shift timestamps."""
    offset_s  = beat_times[0]
    trimmed   = audio[int(offset_s * 1000):]
    shifted   = [t - offset_s for t in beat_times]
    new_dur   = len(trimmed) / 1000.0
    print(f"   ⏱  {label}: snapped {offset_s:.3f}s to first beat  →  {new_dur:.2f}s remaining")
    return shifted, trimmed, new_dur


def match_sample_rate(seg_a: AudioSegment, seg_b: AudioSegment):
    sr = max(seg_a.frame_rate, seg_b.frame_rate)
    ch = max(seg_a.channels,   seg_b.channels)
    if seg_a.frame_rate != sr or seg_a.channels != ch:
        seg_a = seg_a.set_frame_rate(sr).set_channels(ch)
    if seg_b.frame_rate != sr or seg_b.channels != ch:
        seg_b = seg_b.set_frame_rate(sr).set_channels(ch)
    return seg_a, seg_b


# ════════════════════════════════════════════════════════════════════════════
# BEAT REORDERING MODES
# ════════════════════════════════════════════════════════════════════════════

def slice_audio_beats(audio: AudioSegment,
                      beat_times: list, duration: float,
                      label: str = "") -> list:
    boundaries = beat_times + [duration]
    desc = f"✂️  Slicing {label}".strip()
    return [
        audio[int(boundaries[i] * 1000) : int(boundaries[i + 1] * 1000)]
        for i in tqdm(range(len(beat_times)), desc=desc, unit="beat",
                      bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} beats [{elapsed}]")
    ]


def apply_mode(segments: list, mode: str, repeat_times: int = 2) -> list:
    n = len(segments)
    if mode == "remove":
        out = segments[::2]
        print(f"✂️  Kept {len(out)} of {n} beats (removed every other)")

    elif mode == "swap":
        out = []
        for g in range(0, n, 4):
            grp = segments[g:g + 4]
            out += [grp[0], grp[3], grp[2], grp[1]] if len(grp) == 4 else grp
        print(f"🔀 Swapped beats 2 & 4 in every bar ({n} beats)")

    elif mode == "reverse":
        out = list(reversed(segments))
        print(f"⏪ Reversed order of {n} beats")

    elif mode == "shuffle":
        out = segments[:]
        random.shuffle(out)
        print(f"🎲 Shuffled {n} beats randomly")

    elif mode == "repeat":
        out = [seg for seg in segments for _ in range(repeat_times)]
        print(f"🔁 Repeated each of {n} beats ×{repeat_times}  →  {len(out)} total")

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return out


def interleave_beats(segs_a: list, segs_b: list, group: int) -> list:
    """
    Alternate beats from two files.
    group=1 → A B A B ...    group=2 → A A B B A A B B ...
    """
    total = min(len(segs_a), len(segs_b))
    out   = []
    with tqdm(total=total, desc="🔀 Interleaving", unit="beat",
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} beats [{elapsed}]") as pb:
        for i in range(total):
            out.append(segs_a[i] if (i // group) % 2 == 0 else segs_b[i])
            pb.update(1)
    used_a = sum(1 for i in range(total) if (i // group) % 2 == 0)
    print(f"   {total} beats total  ({used_a} from A, {total - used_a} from B)")
    return out


# ════════════════════════════════════════════════════════════════════════════
# MOVIEPY PROGRESS LOGGER
# ════════════════════════════════════════════════════════════════════════════

class FFmpegProgressBar:
    _time_re = re.compile(r"time=(\d+):(\d+):(\d+\.\d+)")

    def __init__(self, total_duration: float):
        self.total = total_duration
        self._pbar = None
        self._last = 0

    def iter_bar(self, **bars):
        name, iterable = next(iter(bars.items()))
        desc = "🔊 Writing audio" if name == "chunk" else f"📦 {name}"
        pb = tqdm(total=len(iterable), desc=desc, unit=name,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]")
        for item in iterable:
            yield item
            pb.update(1)
        pb.close()

    def bars_callback(self, bar, attr, value, old_value=None):
        pass

    def __call__(self, message):
        if self._pbar is None:
            self._pbar = tqdm(
                total=int(self.total), desc="🎬 Exporting video", unit="s",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}s [{elapsed}<{remaining}]",
            )
        m = self._time_re.search(str(message))
        if m:
            cur   = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
            delta = int(cur) - self._last
            if delta > 0:
                self._pbar.update(delta)
                self._last = int(cur)

    def close(self):
        if self._pbar:
            self._pbar.n = self._pbar.total
            self._pbar.refresh()
            self._pbar.close()


# ════════════════════════════════════════════════════════════════════════════
# STATS & PREVIEW
# ════════════════════════════════════════════════════════════════════════════

def print_stats(in_path: str, out_path: str, in_dur: float, out_dur: float):
    in_mb  = os.path.getsize(in_path)  / 1024 / 1024
    out_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n✅ Done!  {out_path}")
    print(f"   Original : {in_dur:.2f}s  ({in_mb:.1f} MB)")
    print(f"   Output   : {out_dur:.2f}s  ({out_mb:.1f} MB)")


def preview(output: str, file_is_video: bool = False):
    """Display an audio player or inline video. Works in Jupyter & Colab."""
    from IPython.display import Audio, HTML, display
    if file_is_video:
        with open(output, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        display(HTML(
            "<p><b>▶ Processed</b></p>"
            '<video width="720" controls>'
            f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
            "</video>"
        ))
    else:
        print("▶ Processed:")
        display(Audio(output))


# ════════════════════════════════════════════════════════════════════════════
# AUDIO PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def process_audio(input_path: str, mode: str, fmt: str,
                  repeat_times: int = 2) -> str:
    wav, dur = extract_wav(input_path)
    beat_times, _, duration = detect_beats(wav)
    os.remove(wav)

    if len(beat_times) < 2:
        raise ValueError("Too few beats detected.")

    audio    = load_audio(input_path)
    segments = slice_audio_beats(audio, beat_times, duration)
    segments = apply_mode(segments, mode, repeat_times)

    with simple_bar("🔗 Stitching") as pb:
        result = sum(segments[1:], segments[0])
        pb.update(1)

    suffix = MODE_SUFFIXES.get(mode, mode)
    out    = make_output_path(input_path, suffix, fmt)

    if fmt == "mp3":
        with simple_bar("💾 Exporting MP3") as pb:
            result.export(out, format="mp3", bitrate="320k")
            pb.update(1)
    else:
        with simple_bar("💾 Exporting FLAC") as pb:
            result.export(out, format="flac", parameters=["-compression_level", "8"])
            pb.update(1)

    print_stats(input_path, out, dur, len(result) / 1000)
    return out


# ════════════════════════════════════════════════════════════════════════════
# VIDEO PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def process_video(input_path: str, mode: str, repeat_times: int = 2) -> str:
    if not _MOVIEPY_AVAILABLE:
        raise ImportError("moviepy is required for video. Install it with: pip install moviepy")

    probe         = probe_video(input_path)
    export_kwargs = build_video_export_params(probe)

    wav, video_dur = extract_wav(input_path)
    beat_times, _, duration = detect_beats(wav)
    os.remove(wav)

    if len(beat_times) < 2:
        raise ValueError("Too few beats detected.")

    boundaries = beat_times + [min(duration, video_dur)]
    video      = VideoFileClip(input_path)

    segments = []
    for i in tqdm(range(len(beat_times)), desc="✂️  Slicing", unit="beat",
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} beats [{elapsed}]"):
        t_s, t_e = boundaries[i], min(boundaries[i + 1], video.duration)
        if t_e - t_s >= 0.01:
            segments.append(video.subclip(t_s, t_e))

    segments = apply_mode(segments, mode, repeat_times)

    with simple_bar("🔗 Concatenating") as pb:
        result = concatenate_videoclips(segments, method="compose")
        pb.update(1)

    suffix = MODE_SUFFIXES.get(mode, mode)
    out    = make_output_path(input_path, suffix, "mp4")

    logger = FFmpegProgressBar(result.duration)
    result.write_videofile(out, logger=logger, **export_kwargs)
    logger.close()

    video.close()
    result.close()

    print_stats(input_path, out, video_dur, result.duration)
    return out


# ════════════════════════════════════════════════════════════════════════════
# INTERLEAVE PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def process_interleave(path_a: str, path_b: str, group: int, fmt: str) -> str:
    print(f"\n🎵 File A : {os.path.basename(path_a)}")
    print(f"🎵 File B : {os.path.basename(path_b)}")
    print(f"   Group  : {group} beat(s) per file before switching")

    print("\n📦 Loading audio files...")
    audio_a = load_audio(path_a)
    audio_b = load_audio(path_b)
    audio_a, audio_b = match_sample_rate(audio_a, audio_b)

    print("\n🔇 Stripping edge silence...")
    audio_a, _ = strip_silence(audio_a, "File A")
    audio_b, _ = strip_silence(audio_b, "File B")

    print()
    wav_a = extract_wav_from_segment(audio_a)
    wav_b = extract_wav_from_segment(audio_b)
    bt_a, bpm_a, dur_a = detect_beats(wav_a, "(File A)")
    bt_b, bpm_b, dur_b = detect_beats(wav_b, "(File B)")
    os.remove(wav_a)
    os.remove(wav_b)

    print("\n🎯 Aligning to first beat...")
    bt_a, audio_a, dur_a = align_to_first_beat(bt_a, audio_a, "File A")
    bt_b, audio_b, dur_b = align_to_first_beat(bt_b, audio_b, "File B")

    bpm_diff = abs(bpm_a - bpm_b)
    if bpm_diff > 5:
        print(f"\n   ⚠️  BPM difference: {bpm_a:.1f} vs {bpm_b:.1f} ({bpm_diff:.1f} apart)")
        print( "      Beat lengths will differ — splices land on beat but durations won't match.")
    else:
        print(f"\n   ✓  BPMs close: {bpm_a:.1f} vs {bpm_b:.1f} ({bpm_diff:.1f} apart) — tight alignment expected")

    segs_a      = slice_audio_beats(audio_a, bt_a, dur_a, "File A")
    segs_b      = slice_audio_beats(audio_b, bt_b, dur_b, "File B")
    interleaved = interleave_beats(segs_a, segs_b, group)

    with simple_bar("🔗 Stitching") as pb:
        result = sum(interleaved[1:], interleaved[0])
        pb.update(1)

    base_b = os.path.basename(os.path.splitext(path_b)[0])
    suffix = f"interleaved_with_{base_b}_group{group}"
    out    = make_output_path(path_a, suffix, fmt)

    if fmt == "mp3":
        with simple_bar("💾 Exporting MP3") as pb:
            result.export(out, format="mp3", bitrate="320k")
            pb.update(1)
    else:
        with simple_bar("💾 Exporting FLAC") as pb:
            result.export(out, format="flac", parameters=["-compression_level", "8"])
            pb.update(1)

    out_mb = os.path.getsize(out) / 1024 / 1024
    print(f"\n✅ Done!  {out}")
    print(f"   Duration : {len(result)/1000:.2f}s  ({out_mb:.1f} MB)")
    return out
