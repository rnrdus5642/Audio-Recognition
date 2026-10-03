"""Diagnose why a recording failed (silent? quiet? ASR limitation?).

Usage:
  python -m python.tools.diagnose_audio recordings/session_XXX/001_사과.wav
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from python.runtime.audio import load_audio_16k_mono


def diagnose(path: Path) -> None:
    audio = load_audio_16k_mono(path)
    duration = len(audio) / 16000
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

    # Silence ratio: how much of the audio is below -40dBFS
    threshold = 10 ** (-40 / 20)  # ~0.01
    silent_frac = float(np.mean(np.abs(audio) < threshold)) if len(audio) else 1.0

    print(f"File: {path}")
    print(f"  Duration:     {duration:.3f}s")
    print(f"  Sample count: {len(audio)}")
    print(f"  RMS:          {rms:.5f}")
    print(f"  Peak:         {peak:.5f}")
    print(f"  Silent ratio: {silent_frac:.1%}  (frames below -40dBFS)")
    print()

    # Diagnosis
    print("DIAGNOSIS:")
    if duration < 0.2:
        print("  ✗ Audio too short (<0.2s). Did the recording really start?")
    if peak < 0.005:
        print("  ✗ COMPLETELY SILENT — mic is not capturing.")
        print("    Likely causes: mic muted / wrong device / permission denied.")
        print("    Fix: --list-devices, then --device <name>, or check")
        print("         Windows mic privacy / volume settings.")
    elif peak < 0.05:
        print("  ⚠ VERY QUIET — mic captures something but at low level.")
        print("    Likely causes: mic boost off / mic far from mouth.")
        print("    Fix: raise mic boost (+10~20dB), speak closer, or pick a")
        print("         different mic (e.g., headset).")
    elif silent_frac > 0.95:
        print("  ⚠ Mostly silent. Speech may be just a brief burst at the")
        print("    edges. Check by listening: start", path)
    elif peak > 0.98:
        print("  ⚠ POSSIBLE CLIPPING — peak at maximum. Lower input gain.")
    else:
        print("  ✓ Audio levels look fine.")
        print("    If ASR still returned empty/wrong, this is a model")
        print("    limitation (see REPORT_PHASE2.md catastrophic ASR cases).")
        print("    Try --pad 300, a longer carrier phrase, or another model.")

    print()
    print("Hint: play the file to confirm what was recorded:")
    print(f"  start {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    args = parser.parse_args()
    if not args.audio.exists():
        print(f"ERROR: file not found: {args.audio}", file=sys.stderr)
        return 1
    diagnose(args.audio)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
