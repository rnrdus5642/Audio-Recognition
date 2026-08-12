"""Recognise once, sweep many times.

Inference is 99% of the cost and none of it depends on the matching
parameters, so this writes the per-frame IPA to disk and every later
sweep replays it in seconds.

Two sets:
  tune   free speech, ages 6-7  - closest to how a child will actually
         talk into the headset
  check  formatted (story reading), age 5 - the only place 5-year-olds
         exist in this corpus, kept out of tuning so it can catch
         choices that only work on older children
"""

import csv
import json
import os
import re
import time

import numpy as np
import soundfile as sf

from python.runtime.recognizer.ko.asr import KoreanASRRecognizer

BASE = os.path.dirname(os.path.abspath(__file__))
SORTED = r"C:\Users\user\Desktop\AIHUB아동오디오데이터\나이별"
OUT = os.path.join(BASE, "tune_frames.json")

WINDOW_S, HOP_S = 2.5, 0.5
PARTICLES = "을를이가와과는은의에도로만부터까지에서으로"

TUNE_POS, TUNE_NEG = 900, 500
CHECK_POS, CHECK_NEG = 350, 250

WORDS = [a["text"] for a in json.load(
    open("shared/targets.json", encoding="utf-8"))["answers"]]


def read(folder):
    path = os.path.join(SORTED, folder, "_목록.csv")
    if not os.path.exists(path):
        return []
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    for r in rows:
        r["folder"] = os.path.join(SORTED, folder)
        text = r["text"]
        r["hits"] = [w for w in WORDS if re.search(
            rf"(^|\s){w}([\s{PARTICLES}]|$|\.|,|\?|!)", text)]
    return rows


def balance(rows, limit):
    """Round-robin over speakers: one child must not carry the number."""
    by = {}
    for r in rows:
        by.setdefault(r["speaker"], []).append(r)
    out, i = [], 0
    while len(out) < limit and any(len(v) > i for v in by.values()):
        for items in by.values():
            if i < len(items) and len(out) < limit:
                out.append(items[i])
        i += 1
    return out


def build(folders, n_pos, n_neg):
    rows = [r for f in folders for r in read(f)]
    pos = balance([r for r in rows if r["hits"]], n_pos)
    neg = balance([r for r in rows if not r["hits"]], n_neg)
    return pos, neg


tune_pos, tune_neg = build(["06세/free", "07세/free"], TUNE_POS, TUNE_NEG)
check_pos, check_neg = build(["05세/formatted"], CHECK_POS, CHECK_NEG)

for label, pos, neg in (("tune (free 6-7세)", tune_pos, tune_neg),
                        ("check (formatted 5세)", check_pos, check_neg)):
    secs = sum(float(r["seconds"] or 0) for r in pos + neg)
    print(f"{label}: 검출 {len(pos)} + 오발동 {len(neg)} · "
          f"{secs/3600:.1f}시간 · 화자 "
          f"{len({r['speaker'] for r in pos + neg})}명", flush=True)

rec = KoreanASRRecognizer()


def frames_of(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    win, hop = int(WINDOW_S * sr), int(HOP_S * sr)
    out, start = [], 0
    while start < len(audio):
        chunk = audio[start:start + win]
        if len(chunk) < win:
            chunk = np.pad(chunk, (win - len(chunk), 0))
        out.append(rec.recognize(chunk))
        start += hop
    return out


result = {"window_s": WINDOW_S, "hop_s": HOP_S, "sets": {}}
t0 = time.time()
total = sum(len(x) for x in (tune_pos, tune_neg, check_pos, check_neg))
done = 0

for name, rows in (("tune_pos", tune_pos), ("tune_neg", tune_neg),
                   ("check_pos", check_pos), ("check_neg", check_neg)):
    items = []
    for r in rows:
        path = os.path.join(r["folder"], r["wav"])
        if not os.path.exists(path):
            continue
        items.append({
            "wav": r["wav"], "age": int(r["age"]), "speaker": r["speaker"],
            "seconds": float(r["seconds"] or 0), "text": r["text"],
            "hits": r["hits"], "frames": frames_of(path),
        })
        done += 1
        if done % 100 == 0:
            print(f"  {done}/{total}  {time.time()-t0:.0f}s", flush=True)
    result["sets"][name] = items

json.dump(result, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\n{time.time()-t0:.0f}초")
print(OUT)
