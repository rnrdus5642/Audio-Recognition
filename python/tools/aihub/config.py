"""Where the corpus lives and which of it we care about.

Every other script in this folder reads these. Change them here rather
than passing paths around.
"""

import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from python.workspace import DATASETS, physical_path

# Wherever the AI Hub download landed. The scripts look for archives
# anywhere underneath, so the vendor's nested folder layout is fine as-is.
RAW = str(physical_path(os.environ.get("AIHUB_RAW", DATASETS / 'AIHUB')))

# Where the tidied corpus is written.
OUT = str(physical_path(os.environ.get("AIHUB_OUT", DATASETS / 'AIHUB정리')))

INDEX = os.path.join(OUT, "index.csv")
_audio_default = Path(OUT) / 'audio' if os.environ.get('AIHUB_OUT') else DATASETS / 'audio'
AUDIO = str(physical_path(os.environ.get('AIHUB_AUDIO', _audio_default)))
SPLITS = os.path.join(OUT, "splits.json")

# Ages to pull audio for. The full corpus holds 738 children aged 5-7
# across 1,028 hours, which is enough on its own - older children were
# going to be borrowed only to make up speaker numbers, and there is
# nothing to make up. Everything else stays in the index only.
AGES = {5, 6, 7}

# Ages the product actually targets - splits are balanced over these.
TARGET_AGES = {5, 6, 7}
