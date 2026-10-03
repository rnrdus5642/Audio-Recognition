"""Interactive live recording + recognition + persistence.

Records from the default microphone, runs the full pipeline (ASR ->
IPA -> matching), shows the result, and saves the audio + a JSON
metadata file. Designed for quickly building a corpus of real
recordings paired with the system's response, useful for:

  * Sanity-checking the pipeline with your own voice
  * Collecting child speech samples for evaluation / fine-tuning
  * Building per-user calibration data

Five recording modes:

  1. Single target, repeat freely:
       python -m python.tools.record_live --target 사과

  2. Walk through a queue of targets from a CSV
     (columns: target_text, target_segment_id[opt]):
       python -m python.tools.record_live --queue practice_words.csv

  3. Walk through the project's own words.csv:
       python -m python.tools.record_live --queue-from-words

  4. Free recording, no target (just probe):
       python -m python.tools.record_live --probe

  5. Custom target on the fly (word not in targets.json):
       python -m python.tools.record_live --target 맥주 --custom-target

By default recordings are saved under:
  recordings/session_<timestamp>/

Each take produces:
  NNN_<answer_id>.wav   - 16 kHz mono PCM_16 audio
  NNN_<answer_id>.json  - full metadata + recognition result

Session summary is written at end:
  session.json
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

# Force UTF-8 stdout on Windows.
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    else:  # pragma: no cover
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer, encoding="utf-8", errors="replace"
        )

import numpy as np

from python.tools.test_real_audio import (
    AudioTester,
    TestResult,
    _print_test_result,
    _result_to_dict,
)
from python.runtime.recognizer.ko import DEFAULT_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS_DIR = PROJECT_ROOT / "recordings"
WORDS_CSV = PROJECT_ROOT / "shared" / "words.csv"

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _input_devices() -> list[tuple[int, dict]]:
    """Return [(device_index, device_info)] for all devices with input channels."""
    import sounddevice as sd

    return [
        (i, d)
        for i, d in enumerate(sd.query_devices())
        if d.get("max_input_channels", 0) > 0
    ]


def _default_input_index() -> int | None:
    import sounddevice as sd

    try:
        default_in = sd.default.device[0]
        return int(default_in) if default_in is not None else None
    except Exception:
        return None


def _list_devices() -> None:
    print("Available input devices (* = default):")
    default_in = _default_input_index()
    for i, d in _input_devices():
        marker = "*" if i == default_in else " "
        name = d.get("name", "?")
        ch = d.get("max_input_channels", 0)
        sr = int(d.get("default_samplerate", 0))
        print(f"  {marker}[{i:>2}] {name:<60s} ch={ch}  sr={sr}")


def _resolve_device(
    spec: str | int | None,
) -> tuple[int | None, str]:
    """Convert user input (int / substring / None) to (device_index, label).

    Rules:
      * None  -> system default, label = "<default: ...>"
      * int   -> exact index match, error if not an input device
      * str numeric -> same as int
      * str other -> case-insensitive substring match against device name
                     If multiple match, the first one (lowest index) wins
                     and the others are listed in a warning.
    """
    import sounddevice as sd

    devices = _input_devices()
    if not devices:
        raise RuntimeError("No input audio devices available")

    # Case 1: no spec -> default
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        idx = _default_input_index()
        if idx is None:
            idx = devices[0][0]
        info = sd.query_devices(idx)
        return idx, f"[default] {info['name']}"

    # Case 2: integer (or numeric string)
    if isinstance(spec, int) or (isinstance(spec, str) and spec.lstrip("-").isdigit()):
        idx = int(spec)
        # Validate
        for i, d in devices:
            if i == idx:
                return idx, d["name"]
        raise ValueError(
            f"Device index {idx} is not an input device. "
            "Run --list-devices to see options."
        )

    # Case 3: substring match
    needle = str(spec).strip().lower()
    matches = [(i, d) for i, d in devices if needle in d["name"].lower()]
    if not matches:
        raise ValueError(
            f"No input device name contains {spec!r}. "
            "Run --list-devices to see options."
        )
    if len(matches) > 1:
        other = ", ".join(
            f"[{i}] {d['name']}" for i, d in matches[1:]
        )
        print(
            f"  (multiple matches for {spec!r}; picking the first. "
            f"Others ignored: {other})"
        )
    idx, info = matches[0]
    return idx, info["name"]


def _pick_device_interactively() -> tuple[int | None, str]:
    """Prompt the user to choose an input device from the list."""
    import sounddevice as sd

    devices = _input_devices()
    if not devices:
        raise RuntimeError("No input audio devices available")
    default_in = _default_input_index()
    print()
    print("Pick an input device:")
    print(f"   [d] system default")
    for i, d in devices:
        marker = "*" if i == default_in else " "
        print(f"  {marker}[{i:>2}] {d['name']:<60s} ch={d['max_input_channels']}")

    while True:
        try:
            choice = input("Device index (or 'd' for default): ").strip().lower()
        except EOFError:
            print("(no input; using default)")
            choice = "d"

        if choice in ("", "d", "default"):
            idx = default_in if default_in is not None else devices[0][0]
            info = sd.query_devices(idx)
            return idx, f"[default] {info['name']}"
        if choice.lstrip("-").isdigit():
            try:
                return _resolve_device(int(choice))
            except ValueError as e:
                print(f"  {e}")
                continue
        # fall through: substring match attempt
        try:
            return _resolve_device(choice)
        except ValueError as e:
            print(f"  {e}")


def record_until_enter(
    device: Optional[int] = None,
    max_seconds: float = 15.0,
    trailing_ms: int = 500,
    pre_roll_ms: int = 200,
) -> np.ndarray:
    """Record until the user presses Enter again (or max_seconds elapses).

    Two buffer adjustments fix the most common "last syllable cut off"
    failure mode:

      * `pre_roll_ms`: how long to keep the mic stream open BEFORE
        announcing "Recording..." so the audio driver is already warm
        when the user starts speaking. Sounddevice's first few
        callbacks after start() can drop ~50ms.

      * `trailing_ms`: how long to keep capturing AFTER the user
        presses Enter. The user often presses Enter just as the last
        syllable ends; the audio thread is ~50-100ms behind real time
        plus 50ms polling latency. 500ms gives plenty of headroom
        without feeling laggy.
    """
    import sounddevice as sd

    chunks: list[np.ndarray] = []

    def _cb(indata, _frames, _time, status):
        if status:
            print(f"  (stream status: {status})", file=sys.stderr)
        chunks.append(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_cb,
        device=device,
    )

    import threading

    stop_event = threading.Event()

    def _wait_for_enter():
        try:
            input()
        except EOFError:
            pass
        stop_event.set()

    with stream:
        # Pre-roll: warm up the driver before the user starts speaking.
        if pre_roll_ms > 0:
            time.sleep(pre_roll_ms / 1000)

        print(
            f"🔴 Recording... (Enter = stop, max {int(max_seconds)}s)",
            flush=True,
        )

        start = time.time()
        t = threading.Thread(target=_wait_for_enter, daemon=True)
        t.start()
        while not stop_event.is_set():
            if time.time() - start > max_seconds:
                print("  (max duration hit; stopping)")
                break
            time.sleep(0.05)

        # Trailing: keep capturing so the last syllable isn't clipped.
        # Most catastrophic ASR-on-short-word failures we saw in Phase 2
        # were actually audio truncation, not the model's fault.
        if trailing_ms > 0:
            time.sleep(trailing_ms / 1000)

    if not chunks:
        return np.zeros(0, dtype=np.float32)
    audio = np.concatenate(chunks, axis=0).flatten().astype(np.float32)
    return audio


def record_fixed(
    duration_s: float, device: Optional[int] = None
) -> np.ndarray:
    """Record for exactly `duration_s` seconds."""
    import sounddevice as sd

    print(f"🔴 Recording for {duration_s:.1f}s...", flush=True)
    audio = sd.rec(
        int(duration_s * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


_SAFE_RE = re.compile(r"[^\w가-힣]+")


def _safe_filename(text: str) -> str:
    cleaned = _SAFE_RE.sub("_", text or "x").strip("_")
    return cleaned or "x"


class RecordingSession:
    """One directory per session, one wav + one json per take."""

    def __init__(
        self,
        base_dir: Path,
        model_name: str,
        matrix_path: Path,
        device_label: str = "",
        device_index: int | None = None,
    ) -> None:
        ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"session_{ts}"
        self.dir = base_dir / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.takes: list[dict] = []
        self.model_name = model_name
        self.matrix_path = matrix_path
        self.device_label = device_label
        self.device_index = device_index

    def add_take(
        self,
        audio: np.ndarray,
        target_text: Optional[str],
        target_segment_id: Optional[str],
        result: TestResult,
        notes: str = "",
    ) -> Path:
        import soundfile as sf

        idx = len(self.takes) + 1
        label = _safe_filename(target_text or "probe")
        stem = f"{idx:03d}_{label}"
        wav_path = self.dir / f"{stem}.wav"
        json_path = self.dir / f"{stem}.json"

        # Save audio
        sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")

        # Audio stats for the metadata
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) if len(audio) else 0.0
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

        # Per-take metadata
        meta = {
            "session_id": self.session_id,
            "take_index": idx,
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "audio_file": wav_path.name,
            "audio_stats": {
                "sample_rate": SAMPLE_RATE,
                "n_samples": int(len(audio)),
                "duration_s": round(len(audio) / SAMPLE_RATE, 3),
                "rms": round(rms, 5),
                "peak": round(peak, 5),
            },
            "input_device": {
                "index": self.device_index,
                "label": self.device_label,
            },
            "target": {
                "text": target_text,
                "segment_id": target_segment_id,
            },
            "model": self.model_name,
            "matrix_path": str(self.matrix_path),
            "result": _result_to_dict(result),
            "notes": notes,
        }
        json_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self.takes.append(meta)
        return wav_path

    def write_summary(self) -> Path:
        summary_path = self.dir / "session.json"
        n_pos = sum(1 for t in self.takes if t["result"].get("overall_pass"))
        n_with_target = sum(
            1 for t in self.takes if t["result"].get("target_text")
        )
        payload = {
            "session_id": self.session_id,
            "model": self.model_name,
            "matrix_path": str(self.matrix_path),
            "input_device": {
                "index": self.device_index,
                "label": self.device_label,
            },
            "started_at": (
                self.takes[0]["timestamp"] if self.takes else None
            ),
            "ended_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "n_takes": len(self.takes),
            "n_with_target": n_with_target,
            "n_passed": n_pos,
            "pass_rate": (
                round(n_pos / n_with_target, 4)
                if n_with_target
                else None
            ),
            "takes": self.takes,
        }
        summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary_path


# ---------------------------------------------------------------------------
# Target queues
# ---------------------------------------------------------------------------


def _load_queue_csv(path: Path) -> list[tuple[str, Optional[str]]]:
    out: list[tuple[str, Optional[str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "target_text" not in (reader.fieldnames or []):
            raise ValueError(
                f"{path} must have a 'target_text' column"
            )
        for raw in reader:
            row = {k: (v or "").strip() for k, v in raw.items()}
            tgt = row["target_text"]
            seg = row.get("target_segment_id") or None
            if tgt:
                out.append((tgt, seg))
    return out


def _load_queue_from_words(path: Path) -> list[tuple[str, Optional[str]]]:
    """Use the project's words.csv. Cycles through every Korean entry."""
    out: list[tuple[str, Optional[str]]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("language") or "").strip() != "ko":
                continue
            text = (row.get("text") or "").strip()
            seg = (row.get("segment_id") or "").strip() or None
            if text:
                out.append((text, seg))
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _print_prompt_for_target(idx: int, target_text: Optional[str]) -> None:
    print()
    print("-" * 72)
    if target_text:
        print(f"[take {idx}]  Say:  「{target_text}」")
    else:
        print(f"[take {idx}]  (probe; no target)")
    print("-" * 72)


def _yesno(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        try:
            ans = input(f"{prompt} {suffix}: ").strip().lower()
        except EOFError:
            return default
        if not ans:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


def _audio_quick_stats(audio: np.ndarray) -> str:
    """Inline one-line stats for sanity-check right after a recording.

    Reports trailing silence specifically because a near-zero trailing
    region is the tell-tale sign of "Enter pressed too early; last
    syllable clipped".
    """
    if len(audio) == 0:
        return "empty"
    duration = len(audio) / SAMPLE_RATE
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    peak = float(np.max(np.abs(audio)))
    # Walk backwards from end of clip to find how much trailing silence
    # there is (frames below -40dBFS).
    threshold = 10 ** (-40 / 20)
    last_loud = len(audio)
    for i in range(len(audio) - 1, -1, -1):
        if abs(audio[i]) >= threshold:
            last_loud = i
            break
    trailing_ms = int((len(audio) - last_loud) / SAMPLE_RATE * 1000)
    warn = ""
    if peak < 0.02:
        warn = "  ⚠ very quiet"
    elif trailing_ms < 80:
        warn = "  ⚠ short trailing silence (may have clipped last sound)"
    elif peak > 0.98:
        warn = "  ⚠ clipping"
    return (
        f"{duration:.2f}s  RMS={rms:.3f}  peak={peak:.3f}  "
        f"trailing_silence={trailing_ms}ms{warn}"
    )


def _process_and_save(
    tester: AudioTester,
    audio: np.ndarray,
    target_text: Optional[str],
    target_segment: Optional[str],
    custom_target: bool,
    session: RecordingSession,
) -> TestResult:
    print(f"  captured: {_audio_quick_stats(audio)}")
    if len(audio) == 0:
        print("WARNING: empty audio buffer (nothing recorded?)")
        # Save anyway so the user has a record of the attempt
        result = TestResult(
            audio_path="<empty>",
            audio_duration_s=0.0,
            hangul="",
            user_ipa=[],
            segment_id=target_segment,
            target_text=target_text,
        )
        session.add_take(
            audio, target_text, target_segment, result, notes="empty buffer"
        )
        return result

    print(
        f"Processing {len(audio) / SAMPLE_RATE:.2f}s of audio "
        "(ASR + matching)..."
    )
    result = tester.test_array(
        audio=audio,
        target_text=target_text,
        segment_id=target_segment,
        custom_target=custom_target,
    )
    # Reuse the rich console formatter from test_real_audio
    _print_test_result(result)

    wav_path = session.add_take(
        audio, target_text, target_segment, result
    )
    print(f"Saved: {wav_path.relative_to(PROJECT_ROOT)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--target",
        help="Target Korean word to record against (e.g., 사과). "
        "Without --queue, you can re-record the same target many times.",
    )
    parser.add_argument("--segment", help="Segment id (optional)")
    parser.add_argument(
        "--custom-target",
        action="store_true",
        help="Treat --target as a brand new word (g2pkk on the fly)",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        help="CSV with target_text [target_segment_id] columns; "
        "walks through one target per take",
    )
    parser.add_argument(
        "--queue-from-words",
        action="store_true",
        help="Walk through every Korean entry in shared/words.csv",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="No target: just record + show ASR + IPA",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="If set, record exactly this many seconds per take "
        "(no Enter-to-stop)",
    )
    parser.add_argument(
        "--trailing-ms",
        type=int,
        default=500,
        help="Extra capture time after Enter (default 500). "
        "Increase if the last syllable is still being clipped.",
    )
    parser.add_argument(
        "--pre-roll-ms",
        type=int,
        default=200,
        help="Driver warm-up time before recording starts "
        "(default 200). Increase if the FIRST syllable is missing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Input device. Accepts either an integer index "
        "(e.g., --device 2) or a case-insensitive name substring "
        "(e.g., --device realtek, --device 'galaxy buds', "
        "--device jbl). Run --list-devices to see available names.",
    )
    parser.add_argument(
        "--pick-device",
        action="store_true",
        help="Show input devices and prompt the user to choose before "
        "starting (overrides --device if both given).",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List input devices and exit",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RECORDINGS_DIR,
        help=f"Where to save sessions (default: {DEFAULT_RECORDINGS_DIR})",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=PROJECT_ROOT
        / "shared"
        / "confusion_matrices"
        / "ko_child_v1.json",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=PROJECT_ROOT / "shared" / "targets.json",
    )
    args = parser.parse_args()

    if args.list_devices:
        _list_devices()
        return 0

    # Validate mode combos
    mode_flags = sum(
        bool(x)
        for x in (
            args.target,
            args.queue,
            args.queue_from_words,
            args.probe,
        )
    )
    if mode_flags == 0:
        parser.error(
            "Choose a mode: --target, --queue, --queue-from-words, or "
            "--probe"
        )
    if mode_flags > 1:
        parser.error(
            "Modes are mutually exclusive: pick one of --target / "
            "--queue / --queue-from-words / --probe"
        )
    if args.custom_target and not args.target:
        parser.error("--custom-target requires --target")

    # Build the queue of targets to record
    queue: list[tuple[Optional[str], Optional[str]]]
    if args.probe:
        # An infinite supply of "no target" prompts; we'll exit on user N
        queue = [(None, None)]
        loop_queue = True
    elif args.target:
        queue = [(args.target, args.segment)]
        loop_queue = True
    elif args.queue:
        items = _load_queue_csv(args.queue)
        if not items:
            parser.error(f"{args.queue} contained no usable rows")
        queue = [(t, s) for t, s in items]
        loop_queue = False
    elif args.queue_from_words:
        items = _load_queue_from_words(WORDS_CSV)
        if not items:
            parser.error(f"{WORDS_CSV} has no Korean entries")
        queue = [(t, s) for t, s in items]
        loop_queue = False
    else:  # unreachable thanks to mode_flags check
        return 1

    # Resolve which microphone to use BEFORE loading the heavy ASR model -
    # if the user picked a bad device name, fail fast.
    try:
        if args.pick_device:
            device_idx, device_label = _pick_device_interactively()
        else:
            device_idx, device_label = _resolve_device(args.device)
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(f"Input device: [{device_idx}] {device_label}")

    print(f"Loading recognizer (model: {args.model})...")
    tester = AudioTester(
        targets_path=args.targets,
        matrix_path=args.matrix,
        model_name=args.model,
    )
    session = RecordingSession(
        base_dir=args.out_dir,
        model_name=args.model,
        matrix_path=args.matrix,
        device_label=device_label,
        device_index=device_idx,
    )
    print(f"Session: {session.dir.relative_to(PROJECT_ROOT)}")

    take_idx = 0
    try:
        if loop_queue:
            # Single target / probe: repeat indefinitely until user quits
            target_text, segment_id = queue[0]
            while True:
                take_idx += 1
                _print_prompt_for_target(take_idx, target_text)
                if args.duration:
                    audio = record_fixed(args.duration, device=device_idx)
                else:
                    input("Press Enter to start recording...")
                    audio = record_until_enter(
                        device=device_idx,
                        trailing_ms=args.trailing_ms,
                        pre_roll_ms=args.pre_roll_ms,
                    )
                _process_and_save(
                    tester=tester,
                    audio=audio,
                    target_text=target_text,
                    target_segment=segment_id,
                    custom_target=args.custom_target,
                    session=session,
                )
                if not _yesno("\nAnother take?"):
                    break
        else:
            # Walk through the queue once
            for target_text, segment_id in queue:
                take_idx += 1
                _print_prompt_for_target(take_idx, target_text)
                if args.duration:
                    audio = record_fixed(args.duration, device=device_idx)
                else:
                    cmd = input(
                        "Enter = record, s = skip, q = quit: "
                    ).strip().lower()
                    if cmd == "q":
                        break
                    if cmd == "s":
                        print("  (skipped)")
                        continue
                    audio = record_until_enter(
                        device=device_idx,
                        trailing_ms=args.trailing_ms,
                        pre_roll_ms=args.pre_roll_ms,
                    )
                _process_and_save(
                    tester=tester,
                    audio=audio,
                    target_text=target_text,
                    target_segment=segment_id,
                    custom_target=False,
                    session=session,
                )
    except KeyboardInterrupt:
        print("\n(interrupted)")

    summary_path = session.write_summary()
    print()
    print("=" * 72)
    print(f"Session complete: {len(session.takes)} take(s)")
    n_with_target = sum(
        1 for t in session.takes if t["result"].get("target_text")
    )
    n_pass = sum(
        1 for t in session.takes if t["result"].get("overall_pass")
    )
    if n_with_target:
        print(
            f"Pass rate: {n_pass}/{n_with_target} "
            f"= {n_pass / n_with_target:.1%}"
        )
    print(f"Summary: {summary_path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
