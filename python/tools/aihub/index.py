"""Build one CSV row per utterance from the label archives.

This is the map of the corpus: age, speaker, length, recording
conditions and transcript for every file. Everything downstream selects
from it rather than walking the audio.

Identity comes from Basic/NumberOfSpeaker, the serial the corpus files
each child's recordings under - not Speaker/SpeakerName, which holds
initials that different children share. In one validation archive alone
110 children answered to 99 names, and five of those names spanned two
ages. Splitting on names would have quietly merged children.

The serial is hashed on the way in. Nothing downstream needs to know
who the children are; the splits only need the same child kept on one
side of the line.

    python python/tools/aihub/index.py
"""

import csv
import hashlib
import json
import os
import sys
import time

from archive import discover, members
from config import INDEX, OUT, RAW

COLUMNS = ["subset", "style", "wav", "age", "gender", "school_year",
           "speaker", "seconds", "speech_start", "speech_end",
           "noise", "device", "environ", "snr", "quality", "text"]


def speaker_hash(name):
    return hashlib.sha256(name.strip().upper().encode()).hexdigest()[:12]


def row(d, archive):
    sp = d.get("Speaker", {})
    env = d.get("Environment", {})
    fi = d.get("File", {})
    misc = d.get("Miscellaneous_Info", {})
    serial = d.get("Basic", {}).get("NumberOfSpeaker", "")
    return {
        "subset": archive.subset,
        "style": archive.style,
        "wav": fi.get("FileName", ""),
        "age": sp.get("Age", ""),
        "gender": sp.get("Gender", ""),
        "school_year": sp.get("SchoolYear", ""),
        "speaker": speaker_hash(serial) if serial else "",
        "seconds": fi.get("FileLength", ""),
        "speech_start": misc.get("SpeechStart", ""),
        "speech_end": misc.get("SpeechEnd", ""),
        "noise": env.get("NoiseEnviron", ""),
        "device": env.get("RecordingDevices", ""),
        "environ": env.get("RecordingEnviron", ""),
        "snr": d.get("Wav", {}).get("SignalToNoiseRatio", ""),
        "quality": d.get("Other", {}).get("QualityStatus", ""),
        "text": d.get("Transcription", {}).get("LabelText", ""),
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else RAW
    labels = [a for a in discover(root) if a.kind == "label"]
    if not labels:
        print(f"라벨 아카이브를 못 찾았습니다: {root}")
        print("원천데이터만 받으셨다면 라벨링데이터도 받아야 합니다.")
        return 1

    print(f"라벨 아카이브 {len(labels)}개")
    for a in labels:
        print(f"  {a}")

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    total, bad = 0, 0
    with open(INDEX, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for a in labels:
            n = 0
            for _m, raw in members(a, ".json"):
                try:
                    d = json.loads(raw.decode("utf-8"))
                except Exception:
                    bad += 1
                    continue
                w.writerow(row(d, a))
                n += 1
                total += 1
                if n % 20000 == 0:
                    print(f"  {a.name} {n:,}  {time.time()-t0:.0f}s",
                          flush=True)
            print(f"  {a.name} 완료 {n:,}건", flush=True)

    print(f"\n{total:,}행" + (f" (실패 {bad})" if bad else "")
          + f"  {time.time()-t0:.0f}초\n{INDEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
