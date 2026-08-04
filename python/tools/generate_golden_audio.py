"""Generate golden test audio for evaluation.

Reads `shared/words.csv`, runs each Korean entry through Edge-TTS with
several voices, and saves the resulting .wav files to
`python/tests/fixtures/audio/`. Also emits `golden_set.json` describing
each case so the evaluation harness knows what target each clip belongs
to.

Edge-TTS produces clean adult speech. This is sufficient for PoC
sanity-checking the pipeline (does our pipeline correctly accept a
clean utterance of the target word? does it correctly reject it when a
*different* word was asked for?). Real
child / developmental-delay speech is out of scope for the golden set
and will be added during real-user testing.

Usage:
    python -m python.tools.generate_golden_audio
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import re
import sys
from pathlib import Path

import edge_tts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORDS_CSV = PROJECT_ROOT / "shared" / "words.csv"
AUDIO_DIR = PROJECT_ROOT / "python" / "tests" / "fixtures" / "audio"
GOLDEN_SET_PATH = (
    PROJECT_ROOT / "python" / "tests" / "fixtures" / "golden_set.json"
)


# Korean Edge-TTS voices used for the golden set.
# Keep it to two for tractable CPU evaluation time.
VOICES = [
    ("ko-KR-SunHiNeural", "f"),
    ("ko-KR-InJoonNeural", "m"),
]


def _read_korean_rows(csv_path: Path) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: (v or "").strip() for k, v in row.items()}
            if row.get("language") == "ko":
                rows.append(row)
    return rows


async def _synthesize(text: str, voice: str, out_path: Path) -> None:
    """Run Edge-TTS and write raw audio to out_path.

    Edge-TTS emits MP3 by default. We collect it in memory, then transcode
    to WAV via librosa+soundfile so the rest of the pipeline (which expects
    16kHz mono WAV) can consume it without changing.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    communicate = edge_tts.Communicate(text, voice)
    mp3_bytes = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_bytes.write(chunk["data"])
    mp3_bytes.seek(0)

    audio, sr = librosa.load(mp3_bytes, sr=16000, mono=True)
    audio = audio.astype(np.float32)
    sf.write(out_path, audio, 16000, subtype="PCM_16")


def _safe_filename(text: str) -> str:
    """Build a filesystem-safe stem from arbitrary text + ascii fallback."""
    keep = re.sub(r"[^\w가-힣]+", "_", text)
    return keep.strip("_") or "x"


async def _run_all(rows: list[dict], out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []

    for row in rows:
        ans_id = row["answer_id"]
        text = row["text"]

        for voice_name, voice_tag in VOICES:
            fname = f"{ans_id}__{voice_tag}.wav"
            out_path = out_dir / fname
            if out_path.exists():
                # Resume support: skip files we've already generated
                print(f"  skip (exists)  {fname}")
            else:
                print(f"  TTS    [{voice_tag}] {text:6s}  -> {fname}")
                await _synthesize(text, voice_name, out_path)

            cases.append(
                {
                    "case_id": f"{ans_id}_{voice_tag}",
                    "audio_path": f"python/tests/fixtures/audio/{fname}",
                    "target_answer_id": ans_id,
                    "target_text": text,
                    "voice": voice_name,
                    "kind": "positive",
                }
            )
    return cases


def _write_golden_set(cases: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "TTS-generated golden set for Phase 2 evaluation. Each case "
            "should be PASSED when scored against the word it says, and "
            "REJECTED when scored against any other word."
        ),
        "voices": [v[0] for v in VOICES],
        "cases": cases,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--words", type=Path, default=WORDS_CSV)
    parser.add_argument("--out-dir", type=Path, default=AUDIO_DIR)
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    args = parser.parse_args()

    if not args.words.exists():
        print(f"ERROR: words.csv not found at {args.words}", file=sys.stderr)
        return 1

    rows = _read_korean_rows(args.words)
    if not rows:
        print("ERROR: no Korean rows in words.csv", file=sys.stderr)
        return 1
    print(
        f"Generating {len(rows) * len(VOICES)} audio files "
        f"({len(rows)} words x {len(VOICES)} voices)"
    )

    cases = asyncio.run(_run_all(rows, args.out_dir))
    _write_golden_set(cases, args.golden_set)
    print(f"\nWrote golden set: {args.golden_set} ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
