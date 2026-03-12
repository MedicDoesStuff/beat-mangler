#!/usr/bin/env python3
"""
cli.py — Beat Mangler command-line interface
Run:  python cli.py
"""

import os
import sys

# Allow running directly from the repo root without installing
sys.path.insert(0, os.path.dirname(__file__))

import beat_mangler as fakers


# ════════════════════════════════════════════════════════════════════════════
# PROMPTS
# ════════════════════════════════════════════════════════════════════════════

def prompt_path(label: str = "your file") -> str:
    while True:
        p = input(f"\n📂 Path to {label}: ").strip().strip("'\"")
        if os.path.isfile(p):
            return p
        print(f"   ✗ File not found: {p!r} — try again.")


def prompt_mode() -> str:
    options = {
        "1": "remove",
        "2": "swap",
        "3": "reverse",
        "4": "shuffle",
        "5": "repeat",
        "6": "interleave",
    }
    print("\n🎛  What do you want to do?")
    print("   1 — Remove every other beat")
    print("   2 — Swap beats 2 & 4 in every bar")
    print("   3 — Reverse beat order (last beat first)")
    print("   4 — Shuffle beats randomly")
    print("   5 — Repeat every beat (stutter)")
    print("   6 — Interleave beats from two files")
    while True:
        choice = input("   Enter 1–6: ").strip()
        if choice in options:
            return options[choice]
        print("   ✗ Please enter a number from 1 to 6.")


def prompt_repeat_times() -> int:
    print("\n🔁 How many times should each beat repeat?")
    print("   e.g. 2 = twice, 3 = three times")
    while True:
        c = input("   Enter a number ≥ 2 (default 2): ").strip() or "2"
        if c.isdigit() and int(c) >= 2:
            return int(c)
        print("   ✗ Please enter a number of 2 or more.")


def prompt_grouping() -> int:
    print("\n🥁 How many consecutive beats from each file before switching?")
    print("   e.g. 1 = A B A B ...  |  2 = A A B B A A B B ...")
    while True:
        c = input("   Enter a number (default 1): ").strip() or "1"
        if c.isdigit() and int(c) >= 1:
            return int(c)
        print("   ✗ Enter a positive integer.")


def prompt_audio_format() -> str:
    print("\n🎵 Output format?")
    print("   1 — MP3  (320 kbps)")
    print("   2 — FLAC")
    while True:
        choice = input("   Enter 1 or 2: ").strip()
        if choice in ("1", "2"):
            return "mp3" if choice == "1" else "flac"
        print("   ✗ Please enter 1 or 2.")


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════╗")
    print("║       🎛  Beat Mangler       ║")
    print("╚══════════════════════════════╝")

    mode = prompt_mode()

    if mode == "interleave":
        path_a = prompt_path("File A (odd beats: 1, 3, 5 ...)")
        path_b = prompt_path("File B (even beats: 2, 4, 6 ...)")
        group  = prompt_grouping()
        fmt    = prompt_audio_format()
        out    = fakers.process_interleave(path_a, path_b, group, fmt)

    else:
        input_path   = prompt_path()
        repeat_times = prompt_repeat_times() if mode == "repeat" else 2
        file_is_vid  = fakers.is_video(input_path)
        audio_fmt    = None if file_is_vid else prompt_audio_format()

        print(f"\n{'🎬' if file_is_vid else '🎵'} Processing: {os.path.basename(input_path)}")
        print(f"   Mode : {mode}" + (f"  ×{repeat_times}" if mode == "repeat" else ""))

        if file_is_vid:
            out = fakers.process_video(input_path, mode, repeat_times)
        else:
            out = fakers.process_audio(input_path, mode, audio_fmt, repeat_times)

    print(f"\n📁 Output: {out}")


if __name__ == "__main__":
    main()
