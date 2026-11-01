#!/usr/bin/env python
"""One command for the organisers:  python run.py --audio-dir <folder> --out <folder>"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from zentro.batch import run_batch  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ZentroMart voice support agent: batch over a folder of customer audio.")
    ap.add_argument("--audio-dir", required=True, help="folder of customer audio files (wav/mp3/ogg/flac/aac/m4a)")
    ap.add_argument("--out", default="outputs", help="where replies (json, text, wav) and retrieved chunks are saved")
    ap.add_argument("--language", default=None, help="STT language code, default from STT_LANGUAGE (en-IN)")
    a = ap.parse_args()
    res = run_batch(a.audio_dir, a.out, a.language)
    print(f"\nDone: {len(res)} files -> {a.out}/ (results.jsonl, summary.csv, *_reply.wav)")
