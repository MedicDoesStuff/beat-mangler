"""
fakers.py - Beat Mangler core library.
Essentia-powered beat manipulation for audio and video.
"""

import base64
import json
import os
import random
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

import essentia.standard as es
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from tqdm.auto import tqdm




# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".flv"}

AUDIO_FMT = {
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

_BAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"

# (ffmpeg codec, display name, preset flag value)
_GPU_CODECS = [
    ("h264_nvenc", "NVIDIA NVENC", "slow"),
    ("h264_amf",   "AMD AMF",     "quality"),
    ("h264_qsv",   "Intel QSV",   "slow"),
]

# hwaccel methods to probe for decoding
_HWACCEL_METHODS = ["cuda", "qsv", "d3d11va", "vaapi", "dxva2"]

_encoder_cache = None
_hwaccel_cache = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _is_video(path):
    return os.path.splitext(path)[1].lower() in VIDEO_EXTS


def _src_fmt(path):
    return AUDIO_FMT.get(os.path.splitext(path)[1].lower().lstrip("."), "mp3")


def _out_path(path, suffix, ext):
    return f"{os.path.splitext(path)[0]}_{suffix}.{ext}"


def _mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def _print_stats(inp, out, in_dur, out_dur):
    print(f"\nDone: {out}")
    print(f"  Input:  {in_dur:.2f}s  ({_mb(inp):.1f} MB)")
    print(f"  Output: {out_dur:.2f}s  ({_mb(out):.1f} MB)")


# ---------------------------------------------------------------------------
# NumPy-based audio I/O (replaces pydub for main pipeline)
# ---------------------------------------------------------------------------

def _load_np(path):
    """Load audio file as float32 numpy array + sample rate."""
    if _is_video(path):
        tmp = tempfile.mktemp(suffix=".wav")
        hwaccel_args = _hwaccel_decode_args()
        cmd = hwaccel_args + [
            "-i", path, "-vn", "-ac", "2", "-ar", "44100",
            "-f", "wav", "-y", tmp,
        ]
        subprocess.run(["ffmpeg"] + cmd, capture_output=True, check=True)
        data, sr = sf.read(tmp, dtype="float32")
        os.remove(tmp)
        return data, sr
    data, sr = sf.read(path, dtype="float32")
    return data, sr


def _load_pydub(path):
    """Load via pydub (only used for silence detection in interleave)."""
    return AudioSegment.from_file(path, format=_src_fmt(path))


def _np_to_mono(samples):
    """Downmix to mono for analysis."""
    if samples.ndim > 1:
        return samples.mean(axis=1).astype(np.float32)
    return samples.astype(np.float32)


def _slice_np(samples, sr, beat_times, duration, label=""):
    """Slice numpy array at beat boundaries — near-zero overhead."""
    bounds = [int(t * sr) for t in beat_times] + [min(int(duration * sr), len(samples))]
    segs = []
    for i in tqdm(range(len(beat_times)),
                  desc=f"  Slicing {label}".rstrip(),
                  unit="beat", bar_format=_BAR):
        segs.append(samples[bounds[i]:bounds[i + 1]])
    return segs


def _stitch_np(segments):
    """O(n) concatenation instead of O(n²) pydub sum."""
    return np.concatenate(segments, axis=0)


def _export_np(samples, sr, path, fmt):
    """Export numpy samples to file. Uses soundfile or ffmpeg for mp3/aac."""
    sf_formats = {"wav", "flac", "ogg", "aiff"}
    if fmt in sf_formats:
        sf.write(path, samples, sr, format=fmt.upper())
    elif fmt in ("mp3", "aac", "m4a"):
        tmp_wav = tempfile.mktemp(suffix=".wav")
        sf.write(tmp_wav, samples, sr, format="WAV")
        codec_map = {"mp3": "libmp3lame", "aac": "aac", "m4a": "aac"}
        codec = codec_map.get(fmt, "libmp3lame")
        ff_fmt = "mp4" if fmt == "m4a" else fmt
        cmd = ["ffmpeg", "-y", "-i", tmp_wav, "-c:a", codec]
        if fmt == "mp3":
            cmd += ["-b:a", "320k"]
        cmd += ["-f", ff_fmt, path]
        subprocess.run(cmd, capture_output=True, check=True)
        os.remove(tmp_wav)
    else:
        sf.write(path, samples, sr)


# ---------------------------------------------------------------------------
# Pydub helpers (only for interleave silence stripping)
# ---------------------------------------------------------------------------

def _pydub_to_np(seg):
    """Convert pydub AudioSegment to numpy float32 array."""
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    if seg.channels > 1:
        samples = samples.reshape(-1, seg.channels)
    samples /= 2 ** (seg.sample_width * 8 - 1)
    return samples, seg.frame_rate


def _np_to_pydub(samples, sr, sample_width=2, channels=None):
    """Convert numpy float32 array to pydub AudioSegment."""
    if channels is None:
        channels = samples.shape[1] if samples.ndim > 1 else 1
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / max(peak, 1.0)
    int_samples = (samples * (2 ** (sample_width * 8 - 1) - 1)).astype(np.int16)
    if int_samples.ndim > 1:
        int_samples = int_samples.flatten()
    return AudioSegment(
        data=int_samples.tobytes(),
        sample_width=sample_width,
        frame_rate=sr,
        channels=channels,
    )


# ---------------------------------------------------------------------------
# GPU encoder detection
# ---------------------------------------------------------------------------

def _test_encoder(codec):
    """Attempt a single-frame encode to verify the encoder works."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i",
             "color=black:s=64x64:d=0.1:r=1",
             "-c:v", codec, "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _detect_encoder():
    """Return (codec, label, preset) for the best available H.264 encoder."""
    global _encoder_cache
    if _encoder_cache is not None:
        return _encoder_cache

    print("  Probing encoders ...", end=" ", flush=True)
    for codec, label, preset in _GPU_CODECS:
        if _test_encoder(codec):
            print(f"{label} ({codec})")
            _encoder_cache = (codec, label, preset)
            return _encoder_cache

    print("no GPU encoder found, using libx264 (CPU)")
    _encoder_cache = ("libx264", "CPU", "slow")
    return _encoder_cache


# ---------------------------------------------------------------------------
# HW-accelerated decoding detection
# ---------------------------------------------------------------------------

def _test_hwaccel(method):
    """Test whether an hwaccel method is available."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-hwaccel", method,
             "-f", "lavfi", "-i", "color=black:s=64x64:d=0.1:r=1",
             "-frames:v", "1", "-f", "null", "-"],
            capture_output=True, timeout=15,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _detect_hwaccel():
    """Detect best available hardware decoding method. Cached."""
    global _hwaccel_cache
    if _hwaccel_cache is not None:
        return _hwaccel_cache

    print("  Probing HW decoders ...", end=" ", flush=True)
    for method in _HWACCEL_METHODS:
        if _test_hwaccel(method):
            print(f"using {method}")
            _hwaccel_cache = method
            return _hwaccel_cache

    print("none found, using CPU decoding")
    _hwaccel_cache = ""
    return _hwaccel_cache


def _hwaccel_decode_args():
    """Return ffmpeg args for hardware-accelerated decoding, or empty list."""
    method = _detect_hwaccel()
    if method:
        return ["-hwaccel", method]
    return []


# ---------------------------------------------------------------------------
# Video probing (ffprobe)
# ---------------------------------------------------------------------------

def _probe_video(path):
    """Extract source video metadata with ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path,
    ]
    info = json.loads(
        subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    )

    streams = info.get("streams", [])
    fmt = info.get("format", {})
    vs = next((s for s in streams if s["codec_type"] == "video"), {})
    aus = next((s for s in streams if s["codec_type"] == "audio"), {})

    num, den = vs.get("r_frame_rate", "30/1").split("/")
    fps = round(int(num) / max(1, int(den)), 3)

    v_br = vs.get("bit_rate")
    a_br = aus.get("bit_rate", "192000")
    if not v_br:
        v_br = str(max(0, int(fmt.get("bit_rate", "0")) - int(a_br)))

    return {
        "width":      int(vs.get("width", 1920)),
        "height":     int(vs.get("height", 1080)),
        "fps":        fps,
        "v_kbps":     max(500, int(v_br) // 1000),
        "pix_fmt":    vs.get("pix_fmt", "yuv420p"),
        "audio_sr":   int(aus.get("sample_rate", 44100)),
        "audio_ch":   int(aus.get("channels", 2)),
        "audio_kbps": max(128, int(a_br) // 1000),
        "duration":   float(fmt.get("duration", 0)),
    }


# ---------------------------------------------------------------------------
# Encoding parameter builders
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# FFmpeg filter_complex concat (single-pass re-encode, frame-accurate)
# ---------------------------------------------------------------------------

def _ffmpeg_concat_reencode(segment_times, input_path, output_path, probe):
    """Cut and concatenate segments in a single ffmpeg pass using filter_complex.

    Stream-copy cannot be used here: beat boundaries almost never land on
    keyframes, so any copy-based cut drifts to the nearest keyframe and
    produces wrong segment lengths/content.  A filter_complex trim+concat
    approach decodes once, trims frame-accurately, and re-encodes once —
    far more reliable than copy and still much faster than multiple passes.
    """
    n = len(segment_times)
    if n == 0:
        return False

    codec, label, preset = _detect_encoder()
    hwaccel_args = _hwaccel_decode_args()
    br      = probe["v_kbps"]
    abr     = min(320, max(128, probe["audio_kbps"]))
    maxrate = int(br * 1.5)
    bufsize = br * 2

    # Build filter_complex:
    #   For each segment i: trim video + audio, reset pts, label outputs.
    #   Then concat all labelled pairs.
    filter_parts = []
    for i, (ts, te) in enumerate(segment_times):
        filter_parts.append(
            f"[0:v]trim=start={ts:.6f}:end={te:.6f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={ts:.6f}:end={te:.6f},asetpts=PTS-STARTPTS[a{i}]"
        )

    v_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_parts.append(f"{v_inputs}concat=n={n}:v=1:a=1[vout][aout]")
    filter_complex = ";".join(filter_parts)

    # Encode params
    if codec == "h264_nvenc":
        enc_args = [
            "-c:v", codec, "-preset", preset,
            "-rc", "vbr", "-b:v", f"{br}k",
            "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
        ]
    elif codec == "h264_amf":
        enc_args = [
            "-c:v", codec, "-quality", preset,
            "-rc", "vbr_peak", "-b:v", f"{br}k",
            "-maxrate", f"{maxrate}k",
        ]
    elif codec == "h264_qsv":
        enc_args = [
            "-c:v", codec, "-preset", preset,
            "-b:v", f"{br}k",
            "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
        ]
    else:
        crf = (16 if br >= 8000 else 18 if br >= 4000 else
               20 if br >= 2000 else 22 if br >= 1000 else 24)
        enc_args = [
            "-c:v", "libx264", "-preset", preset,
            "-crf", str(crf),
            "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
        ]

    cmd = ["ffmpeg", "-y"] + hwaccel_args + ["-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[vout]", "-map", "[aout]",
        "-pix_fmt", probe["pix_fmt"], "-profile:v", "high",
    ] + enc_args + [
        "-c:a", "aac", "-b:a", f"{abr}k",
        "-ar", str(probe["audio_sr"]),
        output_path,
    ]

    print(f"  Encoding with {label} (single-pass filter_complex concat) ...")
    r = subprocess.run(cmd, capture_output=True)

    if r.returncode != 0:
        # If GPU encoder failed, retry with libx264
        if codec != "libx264":
            print(f"  {label} failed, retrying with libx264 ...")
            # Swap out encoder args
            crf = (16 if br >= 8000 else 18 if br >= 4000 else
                   20 if br >= 2000 else 22 if br >= 1000 else 24)
            cpu_enc = [
                "-c:v", "libx264", "-preset", "slow",
                "-crf", str(crf),
                "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
            ]
            cmd2 = ["ffmpeg", "-y", "-i", input_path,
                "-filter_complex", filter_complex,
                "-map", "[vout]", "-map", "[aout]",
                "-pix_fmt", probe["pix_fmt"], "-profile:v", "high",
            ] + cpu_enc + [
                "-c:a", "aac", "-b:a", f"{abr}k",
                "-ar", str(probe["audio_sr"]),
                output_path,
            ]
            r = subprocess.run(cmd2, capture_output=True)

    if r.returncode != 0:
        print(f"  ffmpeg error:\n{r.stderr.decode(errors='replace')[-2000:]}")
        return False

    return True


# ---------------------------------------------------------------------------
# Beat detection
# ---------------------------------------------------------------------------

def _extract_audio_to_wav(path):
    """Extract/convert audio to mono 44100 Hz WAV temp file using ffmpeg with hwaccel."""
    tmp = tempfile.mktemp(suffix=".wav")
    hwaccel_args = _hwaccel_decode_args() if _is_video(path) else []
    cmd = ["ffmpeg", "-y"] + hwaccel_args + [
        "-i", path, "-vn", "-ac", "1", "-ar", "44100",
        "-f", "wav", tmp,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return tmp


def _get_duration(path):
    """Get duration via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", path,
    ]
    info = json.loads(
        subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    )
    return float(info.get("format", {}).get("duration", 0))


def detect_beats(wav_path, label=""):
    tag = f" {label}" if label else ""
    print(f"  Detecting beats{tag} ... ", end="", flush=True)
    audio = es.MonoLoader(filename=wav_path, sampleRate=44100)()
    bt, conf = es.BeatTrackerMultiFeature()(audio)
    dur = len(audio) / 44100.0
    bpm = 60.0 / np.median(np.diff(bt)) if len(bt) > 1 else 0.0
    print(f"{len(bt)} beats, {bpm:.1f} BPM, conf {conf:.3f}")
    return list(bt), bpm, dur


def detect_beats_from_np(samples, sr=44100, label=""):
    """Run beat detection directly on a numpy array (no temp file)."""
    tag = f" {label}" if label else ""
    print(f"  Detecting beats{tag} ... ", end="", flush=True)
    mono = _np_to_mono(samples)
    if sr != 44100:
        mono = es.Resample(inputSampleRate=sr, outputSampleRate=44100)(mono)
        sr = 44100
    bt, conf = es.BeatTrackerMultiFeature()(mono)
    dur = len(mono) / float(sr)
    bpm = 60.0 / np.median(np.diff(bt)) if len(bt) > 1 else 0.0
    print(f"{len(bt)} beats, {bpm:.1f} BPM, conf {conf:.3f}")
    return list(bt), bpm, dur


# ---------------------------------------------------------------------------
# Mode application (works on indices or segments)
# ---------------------------------------------------------------------------

def _apply_mode(segments, mode, repeat_times=2):
    n = len(segments)
    if mode == "remove":
        out = segments[::2]
        print(f"  Kept {len(out)}/{n} beats")
    elif mode == "swap":
        out = []
        for g in range(0, n, 4):
            grp = segments[g:g + 4]
            out += [grp[0], grp[3], grp[2], grp[1]] if len(grp) == 4 else grp
        print(f"  Swapped beats 2 & 4 per bar ({n} beats)")
    elif mode == "reverse":
        out = list(reversed(segments))
        print(f"  Reversed {n} beats")
    elif mode == "shuffle":
        out = segments[:]
        random.shuffle(out)
        print(f"  Shuffled {n} beats")
    elif mode == "repeat":
        out = [s for s in segments for _ in range(repeat_times)]
        print(f"  Repeated {n} beats x{repeat_times} -> {len(out)}")
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return out


def _apply_mode_to_indices(n, mode, repeat_times=2):
    """Apply mode to index list [0..n-1], return reordered indices."""
    indices = list(range(n))
    return _apply_mode(indices, mode, repeat_times)


def _interleave(segs_a, segs_b, group):
    total = min(len(segs_a), len(segs_b))
    out = [
        segs_a[i] if (i // group) % 2 == 0 else segs_b[i]
        for i in range(total)
    ]
    from_a = sum(1 for i in range(total) if (i // group) % 2 == 0)
    print(f"  Interleaved {total} beats ({from_a} A, {total - from_a} B)")
    return out


# ---------------------------------------------------------------------------
# Silence / alignment helpers (interleave)
# ---------------------------------------------------------------------------

def _strip_silence(audio_seg, label="", thresh=-50, chunk=10):
    """Strip silence using pydub (used only in interleave path)."""
    lead = detect_leading_silence(audio_seg, silence_threshold=thresh, chunk_size=chunk)
    trail = detect_leading_silence(audio_seg.reverse(), silence_threshold=thresh, chunk_size=chunk)
    n = len(audio_seg)
    lead = min(lead, n)
    trail = min(trail, n - lead)
    trimmed = audio_seg[lead:n - trail]
    if lead or trail:
        print(f"  {label}: stripped {lead}ms lead + {trail}ms trail")
    return trimmed, lead / 1000.0


def _align_to_beat_np(beat_times, samples, sr, label=""):
    """Align numpy samples to the first beat."""
    off = beat_times[0]
    off_samples = int(off * sr)
    trimmed = samples[off_samples:]
    shifted = [t - off for t in beat_times]
    print(f"  {label}: aligned to first beat at {off:.3f}s")
    return shifted, trimmed, len(trimmed) / float(sr)


def _match_rates_np(a, sr_a, b, sr_b):
    """Match sample rates and channel counts between two numpy arrays."""
    sr = max(sr_a, sr_b)
    if sr_a != sr:
        a = es.Resample(inputSampleRate=sr_a, outputSampleRate=sr)(
            _np_to_mono(a) if a.ndim == 1 else a
        )
        # For stereo, resample each channel
        if a.ndim > 1:
            ch0 = es.Resample(inputSampleRate=sr_a, outputSampleRate=sr)(a[:, 0])
            ch1 = es.Resample(inputSampleRate=sr_a, outputSampleRate=sr)(a[:, 1])
            a = np.column_stack([ch0, ch1])
    if sr_b != sr:
        if b.ndim > 1:
            ch0 = es.Resample(inputSampleRate=sr_b, outputSampleRate=sr)(b[:, 0])
            ch1 = es.Resample(inputSampleRate=sr_b, outputSampleRate=sr)(b[:, 1])
            b = np.column_stack([ch0, ch1])
        else:
            b = es.Resample(inputSampleRate=sr_b, outputSampleRate=sr)(b)

    # Match channels
    if a.ndim != b.ndim:
        if a.ndim == 1:
            a = np.column_stack([a, a])
        if b.ndim == 1:
            b = np.column_stack([b, b])
    elif a.ndim > 1 and a.shape[1] != b.shape[1]:
        ch = max(a.shape[1], b.shape[1])
        if a.shape[1] < ch:
            a = np.column_stack([a] + [a[:, 0:1]] * (ch - a.shape[1]))
        if b.shape[1] < ch:
            b = np.column_stack([b] + [b[:, 0:1]] * (ch - b.shape[1]))

    return a, b, sr



# ---------------------------------------------------------------------------
# Preview (Jupyter / Colab)
# ---------------------------------------------------------------------------

def preview(path):
    from IPython.display import Audio, HTML, display
    if _is_video(path):
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        display(HTML(
            f'<video width="720" controls>'
            f'<source src="data:video/mp4;base64,{data}"></video>'
        ))
    else:
        display(Audio(path))


# ---------------------------------------------------------------------------
# Public: audio pipeline (numpy-based, no redundant loads)
# ---------------------------------------------------------------------------

def process_audio(input_path, mode, fmt, repeat_times=2):
    # Single load
    samples, sr = _load_np(input_path)
    dur = len(samples) / float(sr)

    # Beat detection directly from numpy — no temp file
    bt, _, bdur = detect_beats_from_np(samples, sr)
    if len(bt) < 2:
        raise ValueError("Too few beats detected.")

    # Slice as numpy arrays — near-zero overhead
    segments = _slice_np(samples, sr, bt, bdur)
    segments = _apply_mode(segments, mode, repeat_times)

    # O(n) concatenation
    result = _stitch_np(segments)

    out = _out_path(input_path, MODE_SUFFIXES.get(mode, mode), fmt)
    print(f"  Exporting {fmt} ...")
    _export_np(result, sr, out, fmt)

    out_dur = len(result) / float(sr)
    _print_stats(input_path, out, dur, out_dur)
    return out


# ---------------------------------------------------------------------------
# Public: video pipeline (single-pass filter_complex re-encode)
# ---------------------------------------------------------------------------

def process_video(input_path, mode, repeat_times=2):
    probe = _probe_video(input_path)
    _detect_encoder()       # probe + cache early
    _detect_hwaccel()       # probe + cache early

    # Extract audio for beat detection using ffmpeg with hwaccel
    wav = _extract_audio_to_wav(input_path)
    vdur = probe.get("duration") or _get_duration(input_path)
    bt, _, bdur = detect_beats(wav)
    os.remove(wav)
    if len(bt) < 2:
        raise ValueError("Too few beats detected.")

    # Build time segments
    bounds = bt + [min(bdur, vdur)]
    time_segments = []
    for i in range(len(bt)):
        ts, te = bounds[i], min(bounds[i + 1], vdur)
        if te - ts >= 0.01:
            time_segments.append((ts, te))

    # Apply mode to indices, then reorder time segments
    indices = _apply_mode_to_indices(len(time_segments), mode, repeat_times)
    ordered_segments = [time_segments[i] for i in indices]

    out = _out_path(input_path, MODE_SUFFIXES.get(mode, mode), "mp4")

    if not _ffmpeg_concat_reencode(ordered_segments, input_path, out, probe):
        raise RuntimeError(
            "ffmpeg filter_complex concat failed. "
            "Check that ffmpeg is installed and the input file is valid."
        )

    out_dur = sum(te - ts for ts, te in ordered_segments)
    _print_stats(input_path, out, vdur, out_dur)
    return out


# ---------------------------------------------------------------------------
# Public: interleave pipeline (numpy + parallel beat detection)
# ---------------------------------------------------------------------------

def process_interleave(path_a, path_b, group, fmt):
    print(f"\nFile A: {os.path.basename(path_a)}")
    print(f"File B: {os.path.basename(path_b)}")
    print(f"Group:  {group}")

    # Load both files
    samples_a, sr_a = _load_np(path_a)
    samples_b, sr_b = _load_np(path_b)

    # Match sample rates and channels
    samples_a, samples_b, sr = _match_rates_np(samples_a, sr_a, samples_b, sr_b)

    # Strip silence using pydub (best silence detection)
    seg_a = _np_to_pydub(samples_a, sr)
    seg_b = _np_to_pydub(samples_b, sr)

    seg_a, lead_a = _strip_silence(seg_a, "A")
    seg_b, lead_b = _strip_silence(seg_b, "B")

    # Convert back to numpy after stripping
    samples_a, _ = _pydub_to_np(seg_a)
    samples_b, _ = _pydub_to_np(seg_b)

    # Parallel beat detection — no temp files
    print("  Running parallel beat detection ...")
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_a = ex.submit(detect_beats_from_np, samples_a, sr, "(A)")
        fut_b = ex.submit(detect_beats_from_np, samples_b, sr, "(B)")
        bt_a, bpm_a, dur_a = fut_a.result()
        bt_b, bpm_b, dur_b = fut_b.result()

    # Align to first beat
    bt_a, samples_a, dur_a = _align_to_beat_np(bt_a, samples_a, sr, "A")
    bt_b, samples_b, dur_b = _align_to_beat_np(bt_b, samples_b, sr, "B")

    diff = abs(bpm_a - bpm_b)
    if diff > 5:
        print(f"  Warning: BPM gap {bpm_a:.1f} vs {bpm_b:.1f} ({diff:.1f} apart)")

    # Slice, interleave, stitch — all numpy
    segs_a = _slice_np(samples_a, sr, bt_a, dur_a, "A")
    segs_b = _slice_np(samples_b, sr, bt_b, dur_b, "B")
    interleaved = _interleave(segs_a, segs_b, group)
    result = _stitch_np(interleaved)

    base_b = os.path.splitext(os.path.basename(path_b))[0]
    out = _out_path(path_a, f"interleaved_with_{base_b}_group{group}", fmt)
    print(f"  Exporting {fmt} ...")
    _export_np(result, sr, out, fmt)

    print(f"\nDone: {out}")
    out_dur = len(result) / float(sr)
    print(f"  Duration: {out_dur:.2f}s  ({_mb(out):.1f} MB)")
    return out
