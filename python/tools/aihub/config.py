"""Where the corpus lives and which of it we care about.

Every other script in this folder reads these. Change them here rather
than passing paths around.
"""

import os

# Wherever the AI Hub download landed. The scripts look for archives
# anywhere underneath, so the vendor's nested folder layout is fine as-is.
RAW = os.environ.get("AIHUB_RAW", r"C:\Users\user\Desktop\AIHUB")

# Where the tidied corpus is written.
OUT = os.environ.get("AIHUB_OUT", r"C:\Users\user\Desktop\AIHUB정리")

INDEX = os.path.join(OUT, "index.csv")
AUDIO = os.path.join(OUT, "audio")
SPLITS = os.path.join(OUT, "splits.json")

# Ages to pull audio for. The full corpus holds 738 children aged 5-7
# across 1,028 hours, which is enough on its own - older children were
# going to be borrowed only to make up speaker numbers, and there is
# nothing to make up. Everything else stays in the index only.
AGES = {5, 6, 7}

# Ages the product actually targets - splits are balanced over these.
TARGET_AGES = {5, 6, 7}
