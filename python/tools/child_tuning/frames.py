"""Recognise once, sweep many times.

Inference is 99% of the cost of tuning and none of it depends on the
matching parameters, so the per-frame IPA is written to disk and every
later sweep replays it in seconds. Change the window or the hop and the
cache is void; change a threshold and it is not.

Replaces tune_frames.py, valid_frames.py and more_speakers.py, which
were the same script three times over pointed at hand-picked folders.
Splits now come from tools/aihub, so which children are in which set is
decided in one place rather than here.

Results append as they are produced. The old scripts held everything in
memory until the end, which meant an interrupted run - an hour or more -
left nothing behind.

    python python/tools/child_tuning/frames.py
    python python/tools/child_tuning/frames.py --split test --pos 900

Reading the cache back:

    from frames import load
    items = load("valid")
"""

import argparse
import csv
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "aihub"))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from config import AUDIO, INDEX, OUT, SPLITS  # noqa: E402

FRAMES = os.path.join(OUT, "frames")
WINDOW_S, HOP_S = 2.5, 0.5

# One cache per acoustic model. The cache holds what a particular model
# heard, so a fine-tuned checkpoint and the stock one cannot share it -
# and comparing the two is the entire reason for keeping both.
BASE_TAG = "base"
PARTICLES = "을를이가와과는은의에도로만부터까지에서으로"


def path_of(row):
    return os.path.join(AUDIO, f"{int(row['age']):02d}세",
                        row["style"] or "unknown", row["wav"])


def load(split, tag=BASE_TAG):
    """Every cached utterance for one split, as one model heard it."""
    p = os.path.join(FRAMES, tag, f"{split}.jsonl")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def select(split, words, n_pos, n_neg, isolated=False):
    """Utterances for one split, spread over as many children as possible.

    With `isolated`, detections are restricted to utterances that are the
    word on its own. That is the condition the product actually listens
    in - a child answers a prompt with one word - and it scores
    differently from the same word inside a sentence, because phonemes
    outside the match window cost `skip_cost` and silence produces none.
    Tuning on sentences and shipping for single words is what broke the
    adult thresholds on 2026-08-18.
    """
    speakers = json.load(open(SPLITS, encoding="utf-8"))["speakers"]
    hit = re.compile("|".join(
        rf"(?:(?:^|\s){re.escape(w)}(?:[\s{PARTICLES}]|$|[.,?!]))"
        for w in words))

    pos, neg = [], []
    with open(INDEX, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            who = speakers.get(r["speaker"])
            if who is None or who["split"] != split:
                continue
            r["hits"] = [w for w in words if re.search(
                rf"(^|\s){re.escape(w)}([\s{PARTICLES}]|$|[.,?!])", r["text"])]
            if isolated and r["hits"]:
                if len(re.sub(r"[^가-힣 ]", "", r["text"]).split()) != 1:
                    continue
            (pos if r["hits"] else neg).append(r)

    return balance(pos, n_pos), balance(neg, n_neg)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append",
                    help="기본: train valid test 전부")
    ap.add_argument("--pos", type=int, default=900, help="분할당 검출 케이스")
    ap.add_argument("--neg", type=int, default=600, help="분할당 오발동 케이스")
    ap.add_argument("--model", default=None,
                    help="허브 이름이나 체크포인트 폴더. 기본은 성인 모델")
    ap.add_argument("--isolated", action="store_true",
                    help="검출 케이스를 단독 단어 발화로 제한 (제품 조건)")
    ap.add_argument("--tag", default=None,
                    help="캐시 폴더 이름. 기본은 --model 에서 따옴")
    args = ap.parse_args()
    splits = args.split or ["train", "valid", "test"]
    tag = args.tag or (
        os.path.basename(os.path.normpath(args.model)) if args.model
        else BASE_TAG)

    for p in (INDEX, SPLITS):
        if not os.path.exists(p):
            print(f"없습니다: {p}\n먼저 tools/aihub 를 돌리세요.")
            return 1

    words = [a["text"] for a in json.load(
        open("shared/targets.json", encoding="utf-8"))["answers"]]
    out_dir = os.path.join(FRAMES, tag)
    os.makedirs(out_dir, exist_ok=True)
    print("모델 " + (args.model or "성인 (기본)"))
    print("캐시 " + out_dir)

    plan = {}
    for split in splits:
        pos, neg = select(split, words, args.pos, args.neg, args.isolated)
        rows = [(r, True) for r in pos] + [(r, False) for r in neg]
        rows = [(r, is_pos) for r, is_pos in rows if os.path.exists(path_of(r))]
        plan[split] = rows
        secs = sum(float(r["seconds"] or 0) for r, _ in rows)
        print(f"{split:6} 검출 {len(pos):5,} · 오발동 {len(neg):5,} · "
              f"화자 {len({r['speaker'] for r, _ in rows}):4}명 · "
              f"{secs/3600:5.1f}시간", flush=True)

    todo = sum(len(v) - len({i["wav"] for i in load(s, tag)})
               for s, v in plan.items())
    if not todo:
        print("\n전부 캐시되어 있습니다.")
        return 0

    from python.runtime.recognizer.ko.asr import KoreanASRRecognizer
    rec = (KoreanASRRecognizer(model_name=args.model) if args.model
           else KoreanASRRecognizer())
    rec._load()
    print(f"\n장치: {rec._device}"
          + ("  (torch 가 CPU 빌드입니다 - cu130 을 깔면 훨씬 빠릅니다)"
             if rec._device == "cpu" else ""), flush=True)
    print(f"남은 발화 {todo:,}개", flush=True)

    import numpy as np
    import soundfile as sf

    def frames_of(path):
        """Per-window IPA, and the Hangul it was derived from.

        The recogniser produces both in one acoustic pass and used to
        throw the Hangul away. Keeping it costs nothing and buys two
        things: the cache survives a change to the phonological rules,
        which would otherwise void hours of inference; and a false accept
        can be read back as words rather than as a row of phonemes.
        """
        try:
            audio, sr = sf.read(path, dtype="float32")
        except Exception:
            # About 30 in 125,000 arrive with a broken header. They have a
            # plausible size, so existence is not enough to catch them, and
            # one of them ending a two-hour run is not a good trade.
            return None, None
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        win, hop = int(WINDOW_S * sr), int(HOP_S * sr)
        out, heard, start = [], [], 0
        while start < len(audio):
            chunk = audio[start:start + win]
            if len(chunk) < win:
                chunk = np.pad(chunk, (win - len(chunk), 0))
            text, ipa = rec.recognize_with_text(chunk)
            out.append(ipa)
            heard.append(text)
            start += hop
        return out, heard

    t0, done, skipped = time.time(), 0, 0
    for split, rows in plan.items():
        out_path = os.path.join(out_dir, f"{split}.jsonl")
        have = {i["wav"] for i in load(split, tag)}
        with open(out_path, "a", encoding="utf-8") as fh:
            for r, is_pos in rows:
                if r["wav"] in have:
                    continue
                item = {
                    "wav": r["wav"], "age": int(r["age"]),
                    "speaker": r["speaker"], "style": r["style"],
                    "seconds": float(r["seconds"] or 0), "text": r["text"],
                    "hits": r["hits"], "positive": is_pos,
                }
                item["frames"], item["heard"] = frames_of(path_of(r))
                if item["frames"] is None:
                    skipped += 1
                    continue
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
                fh.flush()
                done += 1
                if done % 100 == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    left = (todo - done) / max(rate, 1e-9)
                    print(f"  {done:,}/{todo:,}  {time.time()-t0:.0f}s"
                          f"  남은 예상 {left/60:.0f}분", flush=True)

    print(f"\n{done:,}개 (건너뜀 {skipped})  {time.time()-t0:.0f}초\n{out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
