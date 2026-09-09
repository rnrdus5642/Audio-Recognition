"""Compare ordinary and music-demo decisions using actual ASR frames or WAV files.

    python -m python.tools.evaluate_music_demo --frames private_frames.json --reference
    python -m python.tools.evaluate_music_demo --audio-csv recordings/evaluation.csv

CSV columns: file,text (optional source). File paths are relative to the CSV.
Each recording is replayed through 2.5 s windows, a 0.5 s hop and 1 s of tail
silence. The report contains aggregate counts, never a child's audio/transcript.
Cached-frame input must contain actual recognition results, not target IPA.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher
from python.tools.test_real_audio import AudioTester

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "shared/confusion_matrices/ko_child_v2.json"
MODEL = ROOT / "python/tools/child_finetune/runs/child/best"
CATALOG = ROOT / "shared/targets_music_demo.json"
STEMS = {
    "xylophone": ["실로폰"], "lift": ["들어", "들기"], "strike": ["쳐", "치기"],
    "put_down": ["놓", "놔"], "tambourine": ["탬버린", "템버린"],
    "shake": ["흔들"], "drum_object": ["북"],
    "drum": ["두드", "두르", "때려", "때리"],
    "grip": ["잡아", "잡기"], "play": ["연주"],
}


def is_negative(label: str, candidates: list[dict]) -> bool:
    """Exclude plausible inflections/phrases from negatives; do not add positives."""
    return not any(stem in label for a in candidates
                   for stem in STEMS[a["action_group"]])


def cached_matcher(matrix):
    """Evaluation-only memoization; all arithmetic remains the runtime's own."""
    matcher = Matcher.for_streaming(matrix)
    score, whole = matcher.score_against, matcher.whole_score
    cached_score = lru_cache(maxsize=8192)(lambda u, t: score(list(u), list(t)))
    cached_whole = lru_cache(maxsize=8192)(lambda u, t: whole(list(u), list(t)))
    matcher.score_against = lambda u, t: cached_score(tuple(u), tuple(t))
    matcher.whole_score = lambda u, t: cached_whole(tuple(u), tuple(t))
    return matcher


def replay(matcher, candidates, frames):
    sm = StreamingMatcher(matcher, candidates)
    for ipa in frames:
        hit = sm.push(ipa)
        if hit:
            return hit
    return None


def normalize_frames(rows):
    return [{"word": r.get("word", r.get("text", "")),
             "source": r.get("source", "recordings"),
             "frames": [f["ipa"] if isinstance(f, dict) else f for f in r["frames"]]}
            for r in rows]


def infer_csv(path: Path, model: str):
    import numpy as np
    from python.runtime.audio import HopFramer, load_audio_16k_mono
    from python.runtime.recognizer.ko.asr import KoreanASRRecognizer
    rec = KoreanASRRecognizer(model_name=model)
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        records = list(csv.DictReader(stream))
    for record in records:
        audio = load_audio_16k_mono(path.parent / record["file"])
        audio = np.pad(audio, (0, (-len(audio)) % 8000 + 16000))
        frames = [rec.recognize(w.samples) for w in HopFramer(40000, 8000).feed(audio)]
        rows.append({"word": record["text"], "source": record.get("source") or "recordings",
                     "frames": frames})
    return rows


def evaluate(rows, catalog, baseline):
    matrix = ConfusionMatrix.from_json(MATRIX)
    matchers = {name: cached_matcher(matrix) for name in ("baseline", "music_demo")}
    current = {a["text"]: a for a in catalog["answers"]}
    pools = {"baseline": baseline, "music_demo": current}
    groups = [("word", w, [w]) for w in current]
    groups += [("stage", str(s["id"]), s["answers"]) for s in catalog["presentation_stages"]]
    report = {}
    for row in rows:
        source = row["source"]
        for kind, key, words in groups:
            cands = [current[w] for w in words]
            label = row["word"]
            positive = label in words
            negative = not positive and is_negative(label, cands)
            if not (positive or negative):
                continue
            for mode in pools:
                stats = report.setdefault(source, {}).setdefault(kind, {}).setdefault(key, {}).setdefault(
                    mode, {"positive_trials": 0, "positive_confirmed": 0,
                           "negative_trials": 0, "negative_confirmed": 0})
                hit = replay(matchers[mode], [pools[mode][w] for w in words], row["frames"])
                prefix = "positive" if positive else "negative"
                stats[prefix + "_trials"] += 1
                stats[prefix + "_confirmed"] += int(hit is not None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", type=Path)
    source.add_argument("--audio-csv", type=Path)
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "recordings/music_demo_report.json")
    args = parser.parse_args()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    tester = AudioTester(targets_path=ROOT / "shared/targets_child.json", matrix_path=MATRIX)
    baseline = {}
    for a in catalog["answers"]:
        # Ordinary web behavior before this profile: configured child words
        # keep their calibration; unknown words use reference-derived marks.
        b = dict(tester._make_custom_target(a["text"])[1])
        b["id"] = a["id"]
        baseline[a["text"]] = b
    if args.frames:
        rows = normalize_frames(json.loads(args.frames.read_text(encoding="utf-8")))
    else:
        rows = infer_csv(args.audio_csv, args.model)
    counts = {s: sum(r["source"] == s for r in rows) for s in sorted({r["source"] for r in rows})}
    if args.reference:
        ref = json.loads(gzip.decompress((ROOT / "shared/reference/reference_frames.json.gz").read_bytes()))
        rows += [{"source": "reference", "word": r["text"], "frames": r["frames"]} for r in ref["items"]]
        counts["reference"] = len(ref["items"])
    result = {
        "profile_id": catalog["profile_id"],
        "catalog_sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "input_kind": "cached_actual_asr_frames" if args.frames else "fresh_audio_inference",
        "recordings": counts, "window_samples": 40000, "hop_samples": 8000,
        "tail_silence_samples": 16000,
        "note": "Tuning/evaluation counts, not independent unseen-child accuracy. Stage trials reuse recordings. Plausible inflections are excluded from negatives.",
        "results": evaluate(rows, catalog, baseline),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for src, grouped in result["results"].items():
        print(src, "clips", counts[src])
        for word, values in grouped["word"].items():
            b, d = values["baseline"], values["music_demo"]
            print(f"  {word}: positive {b['positive_confirmed']} -> {d['positive_confirmed']}/{d['positive_trials']}; "
                  f"negative {b['negative_confirmed']} -> {d['negative_confirmed']}/{d['negative_trials']}")
    print("Report:", args.output)


if __name__ == "__main__":
    main()
