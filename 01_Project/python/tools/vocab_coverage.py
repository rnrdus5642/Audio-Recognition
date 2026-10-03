"""Which curriculum syllables the acoustic model can actually emit.

A CTC head emits exactly `config.vocab_size` classes. A syllable with no
row there cannot appear in a transcript however clearly it is said, so a
word built from one can never be transcribed in its exact form. That is
a property of the model, not of the speaker.

It is not the same as "the word always fails". Matching runs on IPA
through a confusion matrix, so a near-miss transcript can still score
above threshold, and a word can be confirmed without ever being spelled
correctly. Read a hit here as "measure this word before shipping it",
not as a verdict on it.

The tokenizer is not the model. `len(tokenizer)` counts sentence markers
the head has no row for, so taking ids straight out of vocab.json
overstates what can be emitted by however many extras a checkpoint
happens to carry. Everything below is bounded by the acoustic width and
says so when the two disagree.

    python -m python.tools.vocab_coverage
    python -m python.tools.vocab_coverage --vocab python/tools/child_finetune/runs/child/best
    python -m python.tools.vocab_coverage --strict --allow N --json report.json
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STREAMING = PROJECT_ROOT / "unity" / "Assets" / "StreamingAssets"
SAMPLES = (PROJECT_ROOT / "unity" / "Packages"
           / "com.domicube.phoneme-matching" / "Samples~" / "KoreanData")
SHARED = PROJECT_ROOT / "shared"

DEFAULT_VOCAB = STREAMING / "wav2vec2_ko_vocab.json"

# The committed, shipping copies. shared/targets*.json is a build
# artifact and gitignored, so auditing it by default would pass here and
# find nothing at all on a fresh clone.
DEFAULT_SOURCES = (
    STREAMING / "targets.json",
    STREAMING / "targets_child.json",
    SAMPLES / "targets.json",
    SAMPLES / "targets_child.json",
    SHARED / "words.csv",
)

GENERATED_SOURCES = (
    SHARED / "targets.json",
    SHARED / "targets_child.json",
    SHARED / "targets_norules.json",
)

HANGUL_FIRST, HANGUL_LAST = "\uac00", "\ud7a3"


class VocabularyError(ValueError):
    """A vocabulary disagrees with the model it claims to describe."""


@dataclass(frozen=True)
class Vocabulary:
    """The tokens a CTC head can emit, and the extras it cannot.

    `size` is the acoustic width - the number of logits per frame. `ids`
    only ever holds tokens inside it. `beyond` holds what the tokenizer
    knows about anyway, which is where the sentence markers live and why
    `len(tokenizer)` is the wrong number to reach for.
    """

    source: str
    size: int
    ids: dict[str, int]
    beyond: dict[str, int] = field(default_factory=dict)
    blank_id: int | None = None
    unk_id: int | None = None
    delimiter_id: int | None = None
    # Where `size` came from. "config" and "export" trace back to a model;
    # "declared" is a number the caller supplied, which is a claim and not
    # proof; "assumed" is the widest id, which counts tokenizer-only
    # markers the head may have no column for.
    width_source: str = "config"

    def emits(self, token: str) -> bool:
        return token in self.ids

    @property
    def width_verified(self) -> bool:
        return self.width_source in ("config", "export")

    def describe(self) -> dict:
        return {
            "source": self.source,
            "size": self.size,
            "width_source": self.width_source,
            "width_verified": self.width_verified,
            "blank_id": self.blank_id,
            "unk_id": self.unk_id,
            "delimiter_id": self.delimiter_id,
            "beyond_acoustic_range": dict(sorted(
                self.beyond.items(), key=lambda kv: kv[1])),
        }


def _check_token_forms(base: dict[str, int], source: str) -> None:
    """Refuse a vocabulary whose tokens are not already in NFC.

    Target text is normalised to NFC before it is compared, so a token
    stored decomposed would never match and the word needing it would be
    reported as supported. Normalising the token here instead would
    change what an id means, and where both forms are present it would
    merge two ids onto one syllable. Neither is this tool's call to make,
    so it refuses rather than repairs.
    """
    odd = {t: unicodedata.normalize("NFC", t) for t in base
           if unicodedata.normalize("NFC", t) != t}
    if not odd:
        return
    both = sorted(nfc for t, nfc in odd.items() if nfc in base)
    detail = f"; 이미 NFC 형태도 있음: {both[:3]}" if both else ""
    raise VocabularyError(
        f"{source}: NFC 가 아닌 토큰 {len(odd)}개 "
        f"{sorted(odd)[:3]!r}{detail}")


def _content(value, field_name: str, source: str) -> str | None:
    """A special token may be a bare string or an AddedToken mapping."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("content"), str):
        return value["content"]
    raise VocabularyError(
        f"{source}: {field_name} 를 토큰으로 읽을 수 없습니다: {value!r}")


def _special_id(name: str, settings: dict, known: dict[str, int], size: int,
                source: str) -> int | None:
    """Resolve a declared special token, or say it was never declared.

    A missing entry stays None. A declared one that no id backs, or one
    the head cannot emit, is a fault in the checkpoint rather than
    something to guess around.
    """
    token = _content(settings.get(name), name, source)
    if token is None:
        return None
    if token not in known:
        raise VocabularyError(
            f"{source}: {name} {token!r} 이 어휘에 없습니다")
    if known[token] >= size:
        raise VocabularyError(
            f"{source}: {name} {token!r} 의 id {known[token]} 가 음향 범위 "
            f"밖입니다 (< {size} 이어야 함)")
    return known[token]


def _reconcile(base: dict[str, int], extra: dict[str, int],
               source: str) -> dict[str, int]:
    """Fold the added-token table into the base map, or refuse to.

    Checkpoints mirror `[PAD]`, `[UNK]` and the delimiter in both tables.
    Repeating a token at the id it already has is redundant and fine. The
    same token at two ids, or two tokens claiming one id, is not: one of
    the two readings has to be wrong and picking one silently is how an
    off-by-one vocabulary ships.
    """
    resolved: dict[str, int] = {}
    for token, i in extra.items():
        if token in base:
            if base[token] != i:
                raise VocabularyError(
                    f"{source}: {token!r} 이 어휘에서는 {base[token]}, "
                    f"추가 토큰에서는 {i} 입니다")
            continue
        resolved[token] = i

    taken = {i: t for t, i in base.items()}
    for token, i in resolved.items():
        if i in taken:
            raise VocabularyError(
                f"{source}: id {i} 를 {taken[i]!r} 와 {token!r} 이 함께 "
                "차지합니다")
        taken[i] = token
    return resolved


def _build(base: dict[str, int], tokenizer_only: dict[str, int], size: int,
           source: str, width_source: str = "config",
           **named: int | None) -> Vocabulary:
    """Check a token map against a declared acoustic width.

    Three ways this goes wrong quietly, so all three are refused rather
    than reported: two tokens sharing an id, which makes the decoder pick
    one arbitrarily; a gap below the width, which leaves a logit column
    decoding to nothing; and a tokenizer-only marker sitting inside the
    width, which would let the head emit a sentence marker into a
    transcript.
    """
    _check_token_forms(base, source)
    by_id: dict[int, str] = {}
    for token, i in sorted(base.items(), key=lambda kv: kv[1]):
        if i in by_id:
            raise VocabularyError(
                f"{source}: id {i} is used by both {by_id[i]} and {token}")
        by_id[i] = token

    # Before the holes, because a marker that has taken an acoustic id is
    # both the more specific fault and the reason for the hole it leaves.
    collisions = {t: i for t, i in tokenizer_only.items() if i < size}
    if collisions:
        raise VocabularyError(
            f"{source}: tokenizer-only token(s) {sorted(collisions)} sit "
            f"inside the acoustic range (< {size}); the head would be able "
            "to emit them")

    holes = [i for i in range(size) if i not in by_id]
    if holes:
        raise VocabularyError(
            f"{source}: no token for id(s) {holes[:5]} below the acoustic "
            f"width {size}")

    duplicated = sorted(set(base) & set(tokenizer_only))
    if duplicated:
        raise VocabularyError(
            f"{source}: {duplicated} appear both in the vocabulary and in "
            "the added tokens")

    beyond = {t: i for t, i in base.items() if i >= size}
    beyond.update(tokenizer_only)
    return Vocabulary(
        source=source, size=size,
        ids={t: i for t, i in base.items() if i < size},
        beyond=beyond, width_source=width_source, **named)


def load_exported(path: Path) -> Vocabulary:
    """Read the Unity-facing export from `python.tools.export_ctc_vocab`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tokens = data["tokens"]
    size = int(data.get("vocab_size", len(tokens)))
    if size != len(tokens):
        raise VocabularyError(
            f"{path}: declares vocab_size {size} but lists {len(tokens)} "
            "tokens")
    return _build(
        {t: i for i, t in enumerate(tokens)}, {}, size, str(path),
        width_source="export",
        blank_id=data.get("blank_id"), unk_id=data.get("unk_id"),
        delimiter_id=data.get("word_delimiter_id"))


def load_checkpoint(directory: Path, size: int | None = None) -> Vocabulary:
    """Read a Hugging Face checkpoint's vocabulary, bounded by its config.

    `config.json` is what decides the width. vocab.json on its own counts
    whatever the tokenizer carries, which is a different and larger
    number on every checkpoint here.

    Without a config there is nothing to bound it, and guessing from the
    widest id would count the sentence markers as emittable. Pass `size`
    or point this at a checkpoint that carries its config.
    """
    directory = Path(directory)
    base = {t: int(i) for t, i in json.loads(
        (directory / "vocab.json").read_text(encoding="utf-8")).items()}

    config_path = directory / "config.json"
    width_source = "declared"
    if size is not None:
        pass
    elif config_path.exists():
        size = int(json.loads(
            config_path.read_text(encoding="utf-8"))["vocab_size"])
        width_source = "config"
    else:
        raise VocabularyError(
            f"{directory}: config.json 이 없어 음향 폭을 확인할 수 없습니다. "
            "--acoustic-size 로 직접 주거나 config 가 있는 체크포인트를 "
            "가리키세요")

    settings: dict = {}
    tokenizer_config = directory / "tokenizer_config.json"
    if tokenizer_config.exists():
        settings = json.loads(tokenizer_config.read_text(encoding="utf-8"))

    extra: dict[str, int] = {}
    added = directory / "added_tokens.json"
    if added.exists():
        extra = {t: int(i) for t, i in
                 json.loads(added.read_text(encoding="utf-8")).items()}
    for i, token in settings.get("added_tokens_decoder", {}).items():
        content = _content(token, f"added_tokens_decoder[{i}]",
                           str(directory))
        if content in extra and extra[content] != int(i):
            raise VocabularyError(
                f"{directory}: {content!r} 이 added_tokens.json 에서는 "
                f"{extra[content]}, tokenizer_config 에서는 {i} 입니다")
        extra[content] = int(i)

    extra = _reconcile(base, extra, str(directory))
    known = dict(base)
    known.update(extra)
    return _build(
        base, extra, size, str(directory), width_source=width_source,
        blank_id=_special_id("pad_token", settings, known, size,
                             str(directory)),
        unk_id=_special_id("unk_token", settings, known, size,
                           str(directory)),
        delimiter_id=_special_id("word_delimiter_token", settings, known,
                                 size, str(directory)))


def load_vocabulary(path: Path, size: int | None = None) -> Vocabulary:
    """Accept a checkpoint directory, an export, or a bare vocab.json."""
    path = Path(path)
    if path.is_dir():
        return load_checkpoint(path, size)
    if path.name == "vocab.json" and (path.parent / "config.json").exists():
        return load_checkpoint(path.parent, size)

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "tokens" in data:
        return load_exported(path)
    if not isinstance(data, dict) or not data or not all(
            isinstance(v, int) for v in data.values()):
        raise VocabularyError(f"{path}: not a vocabulary this can read")
    if size is not None:
        return _build(data, {}, size, str(path), width_source="declared")
    # No config, so nothing has checked this against a model. The widest
    # id stands in for the width and every report carries
    # width_verified=False, because that number counts any tokenizer-only
    # marker the head has no column for.
    return _build(data, {}, max(data.values()) + 1,
                  f"{path} (음향 폭 미확인: 가장 큰 id 로 가정)",
                  width_source="assumed")


@dataclass(frozen=True)
class Entry:
    """One answer to check, and where it came from."""

    source: str
    word_id: str
    text: str


def syllables(text: str) -> list[str]:
    """The Hangul syllables of a target, in order, everything else gone."""
    # Targets arrive from JSON and CSV that other tools have edited, so a
    # decomposed 놔 (U+1102 U+116A) is possible. Left alone it selects no
    # syllables at all and the word reads as fully supported. Composing
    # is safe here in a way it is not for token keys: this is the text
    # being asked about, not a mapping anything decodes through.
    return [c for c in unicodedata.normalize("NFC", text)
            if HANGUL_FIRST <= c <= HANGUL_LAST]


def read_targets(path: Path, label: str) -> list[Entry]:
    """Answers out of a targets file, in either shape it comes in.

    2.0.0 is a flat list of answers. 1.0.0 grouped them into segments,
    and `targets_norules.json` is still that shape. Anything else raises:
    a targets file this cannot read must not come back as an empty list,
    because empty reads as full coverage.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "answers" in data:
        answers = data["answers"]
    elif isinstance(data, dict) and "segments" in data:
        answers = [a for segment in data["segments"]
                   for a in segment.get("answers", [])]
    else:
        raise VocabularyError(
            f"{path}: answers 도 segments 도 없습니다. targets 파일이 "
            "맞습니까?")

    missing = [a for a in answers if not isinstance(a, dict) or "text" not in a]
    if missing or not answers:
        raise VocabularyError(
            f"{path}: text 가 없는 항목 {len(missing)}개 / 전체 "
            f"{len(answers)}개")
    return [Entry(label, a.get("id", ""), a["text"]) for a in answers]


def read_words(path: Path, label: str) -> list[Entry]:
    """words.csv, read the way `build_targets` reads it.

    Deliberately not imported from there: that module pulls in the whole
    G2P stack, and a check that has to load mecab before it can tell you
    a syllable is missing is a check nobody runs before adding a word.
    """
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = Path(path).read_text(encoding="cp949")
    with io.StringIO(text, newline="") as handle:
        return [Entry(label, (row.get("answer_id") or "").strip(),
                      row["text"].strip())
                for row in csv.DictReader(handle)
                if (row.get("text") or "").strip()]


def read_source(path: Path) -> list[Entry]:
    """Label a source by enough of its path to tell the copies apart."""
    path = Path(path)
    # targets.json exists in three places with the same filename; the
    # containing folder is what says which one an entry came from.
    label = f"{path.parent.name}/{path.name}"
    if path.suffix == ".csv":
        return read_words(path, label)
    return read_targets(path, label)


def audit(entries: list[Entry], vocab: Vocabulary) -> dict:
    """Report every answer holding a syllable the head cannot emit."""
    unsupported: dict[str, dict] = {}
    answers: list[dict] = []
    per_source: dict[str, dict] = {}
    seen: set[str] = set()

    for entry in entries:
        summary = per_source.setdefault(
            entry.source,
            {"source": entry.source, "answers": 0, "unsupported_answers": 0})
        summary["answers"] += 1

        missing: list[str] = []
        for syllable in syllables(entry.text):
            seen.add(syllable)
            if not vocab.emits(syllable) and syllable not in missing:
                missing.append(syllable)
        if not missing:
            continue

        summary["unsupported_answers"] += 1
        answers.append({"source": entry.source, "id": entry.word_id,
                        "text": entry.text, "missing": missing})
        for syllable in missing:
            record = unsupported.setdefault(
                syllable, {"answers": [], "texts": []})
            record["answers"].append(f"{entry.source}:{entry.word_id}")
            if entry.text not in record["texts"]:
                record["texts"].append(entry.text)

    return {
        "vocabulary": vocab.describe(),
        "checked": {"answers": len(entries), "unique_syllables": len(seen),
                    "sources": len(per_source)},
        "sources": list(per_source.values()),
        "unsupported": dict(sorted(unsupported.items())),
        "unsupported_answers": answers,
        "_note": ("어휘에 없으면 그 단어를 정확히 전사할 수 없다는 뜻입니다. "
                  "매칭은 IPA 와 혼동행렬로 하므로 확정이 항상 불가능하다는 "
                  "뜻은 아니며, 실제 확정률은 녹음으로 재야 합니다."),
    }


def format_report(report: dict) -> str:
    vocab = report["vocabulary"]
    lines = [
        f"어휘 {vocab['source']}",
        f"  음향 출력 {vocab['size']}개  (blank {vocab['blank_id']} · "
        f"unk {vocab['unk_id']} · 구분자 {vocab['delimiter_id']})",
    ]
    if not vocab["width_verified"]:
        lines.append(f"  주의: 이 폭의 출처는 {vocab['width_source']} 이며 "
                     "모델 config 로 확인된 값이 아닙니다")
    if vocab["beyond_acoustic_range"]:
        pairs = "  ".join(f"{t} {i}" for t, i
                          in vocab["beyond_acoustic_range"].items())
        lines.append(f"  출력 밖 (토크나이저 전용): {pairs}")

    lines.append("")
    for summary in report["sources"]:
        state = (f"낼 수 없는 단어 {summary['unsupported_answers']}개"
                 if summary["unsupported_answers"] else "전부 가능")
        lines.append(f"  {summary['source']:34} "
                     f"{summary['answers']:4}단어   {state}")

    lines.append("")
    if not report["unsupported"]:
        lines.append("모든 음절이 어휘 안에 있습니다.")
        return "\n".join(lines)

    lines.append(f"어휘에 없는 음절 {len(report['unsupported'])}개")
    for syllable, record in report["unsupported"].items():
        lines.append(f"  {syllable}   {' · '.join(record['texts'])}"
                     f"   ({len(record['answers'])}곳)")
    lines.append("")
    lines.append("정확한 전사가 불가능하다는 뜻입니다. 매칭은 IPA 로 하므로")
    lines.append("확정이 항상 불가능하다는 뜻은 아니며, 실제 확정률은 해당")
    lines.append("단어 녹음으로 재야 합니다.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # Before parsing, so --help and argparse's own errors are readable
    # too. This is the CLI entry point; importing the module leaves the
    # caller's streams alone.
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB,
                   help="체크포인트 디렉터리, export_ctc_vocab 산출물, "
                        "또는 vocab.json")
    p.add_argument("--acoustic-size", type=int, default=None,
                   help="config.json 이 없을 때 음향 출력 폭을 직접 지정")
    p.add_argument("--source", type=Path, action="append", default=None,
                   help="검사할 targets.json 이나 words.csv (여러 번 가능)")
    p.add_argument("--include-generated", action="store_true",
                   help="빌드 산출물인 shared/targets*.json 도 검사")
    p.add_argument("--json", type=Path, default=None,
                   help="보고서를 이 경로에 기록")
    p.add_argument("--allow", action="append", default=[],
                   help="이미 알고 있어 --strict 를 깨지 않을 음절")
    p.add_argument("--strict", action="store_true",
                   help="--allow 밖의 음절이 있으면 실패로 끝냄")
    args = p.parse_args(argv)

    paths = list(args.source) if args.source else list(DEFAULT_SOURCES)
    if args.include_generated and not args.source:
        paths += [path for path in GENERATED_SOURCES if path.exists()]

    missing = [path for path in paths if not path.exists()]
    if missing:
        print(f"ERROR: 없는 파일 {missing[0]}", file=sys.stderr)
        return 1

    try:
        vocab = load_vocabulary(args.vocab, args.acoustic_size)
        entries: list[Entry] = []
        for path in paths:
            entries.extend(read_source(path))
    except (VocabularyError, KeyError, ValueError, OSError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    report = audit(entries, vocab)
    print(format_report(report))

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"\n→ {args.json}")

    allowed = {unicodedata.normalize("NFC", s) for s in args.allow}
    unexpected = sorted(set(report["unsupported"]) - allowed)
    if args.strict and unexpected:
        print(f"\nERROR: 예상 밖의 음절 {unexpected}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
