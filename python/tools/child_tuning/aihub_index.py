"""Index every label so audio can be pulled by age.

Writes one CSV row per utterance. Reading 274k tiny JSON files takes
about half an hour on Windows, so this runs once and everything after it
reads the CSV instead.
"""

import csv
import glob
import json
import os
import time

ROOT = (r"C:\Users\user\Desktop\AIHUB아동오디오데이터"
        r"\011.한국어 아동 음성 데이터\01.데이터\2.Validation\라벨링데이터")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aihub_index.csv")

rows = 0
t0 = time.time()
with open(OUT, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["set", "wav", "age", "gender", "school_year", "speaker",
                "seconds", "noise", "device", "environ", "text"])

    for sub, name in (("_x_free", "free"), ("_x_formatted", "formatted")):
        files = glob.glob(os.path.join(ROOT, sub, "**", "*.json"), recursive=True)
        print(f"{name}: {len(files):,}개", flush=True)
        for n, f in enumerate(files, 1):
            try:
                d = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            sp = d.get("Speaker", {})
            env = d.get("Environment", {})
            fi = d.get("File", {})
            w.writerow([
                name,
                fi.get("FileName", ""),
                sp.get("Age", ""),
                sp.get("Gender", ""),
                sp.get("SchoolYear", ""),
                sp.get("SpeakerName", ""),
                fi.get("FileLength", ""),
                env.get("NoiseEnviron", ""),
                env.get("RecordingDevices", ""),
                env.get("RecordingEnviron", ""),
                d.get("Transcription", {}).get("LabelText", ""),
            ])
            rows += 1
            if n % 20000 == 0:
                print(f"  {name} {n:,}/{len(files):,}  {time.time()-t0:.0f}s",
                      flush=True)

print(f"\n{rows:,}행  {time.time()-t0:.0f}초")
print(OUT)
