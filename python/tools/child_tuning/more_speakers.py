"""Add formatted 6-7 year olds to the tuning set.

Fifteen children is too few to fit a threshold of 0.925 to. The corpus
has 105 more in the same age range - reading stories rather than talking
freely, so the speech is less natural, but speaker variety is what the
tuning set is short of, not naturalness.
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
OUT = os.path.join(BASE, "more_frames.json")

WINDOW_S, HOP_S = 2.5, 0.5
PARTICLES = "을를이가와과는은의에도로만부터까지에서으로"
N_POS, N_NEG = 900, 600

WORDS = [a["text"] for a in json.load(
    open("shared/targets.json", encoding="utf-8"))["answers"]]


def read(folder):
    rows = list(csv.DictReader(
        open(os.path.join(SORTED, folder, "_목록.csv"), encoding="utf-8-sig")))
    for r in rows:
        r["folder"] = os.path.join(SORTED, folder)
        r["hits"] = [w for w in WORDS if re.search(
            rf"(^|\s){w}([\s{PARTICLES}]|$|\.|,|\?|!)", r["text"])]
    return rows


def balance(rows, limit):
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


rows = read("06세/formatted") + read("07세/formatted")
pos = balance([r for r in rows if r["hits"]], N_POS)
neg = balance([r for r in rows if not r["hits"]], N_NEG)
print(f"검출 {len(pos)} + 오발동 {len(neg)} · 화자 "
      f"{len({r['speaker'] for r in pos + neg})}명 · "
      f"{sum(float(r['seconds'] or 0) for r in pos+neg)/3600:.1f}시간", flush=True)

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
t0, done = time.time(), 0
total = len(pos) + len(neg)
for name, rows_ in (("more_pos", pos), ("more_neg", neg)):
    items = []
    for r in rows_:
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
print(f"\n{time.time()-t0:.0f}초\n{OUT}", flush=True)
