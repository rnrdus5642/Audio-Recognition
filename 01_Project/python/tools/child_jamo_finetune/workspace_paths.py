"""Workspace helpers for tools run directly as Python scripts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from python.workspace import (  # noqa: E402,F401
    DATASETS, PROJECT, RECORDINGS, WORKSPACE, historical_path, physical_path,
)
