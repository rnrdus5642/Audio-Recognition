"""Build the opt-in music demonstration catalog for Python and Unity.

    python -m python.build.build_music_presentation

Thresholds and word confirmation policies live in shared/music_presentation.json.
This build uses the ordinary G2P and action comparisons. No recorded templates
or alternative recognizer are loaded by the presentation runtime.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .build_targets import (
    WordRow, build_answer_entries, read_words_csv, add_contrastive_alternatives,
)
from .g2p import get_g2p
from python.runtime.matching.matcher import validate_catalog

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "shared/music_presentation.json"
PACKAGE = ROOT / "unity/Packages/com.domicube.phoneme-matching"
BUNDLED_DATA = PACKAGE / "Runtime/Resources/DomiCube/PhonemeMatching/MusicDemo"
FEEDBACK_DATA = PACKAGE / "Runtime/Resources/DomiCube/PhonemeMatching/Feedback"
DESTINATIONS = [
    ROOT / "shared/targets_music_demo.json",
    ROOT / "unity/Assets/StreamingAssets/targets_music_demo.json",
    ROOT / "unity/Packages/com.domicube.phoneme-matching/Samples~/KoreanData/targets_music_demo.json",
    BUNDLED_DATA / "targets_music_demo.json",
]


def build_catalog(config: dict) -> dict:
    words = config["words"]
    ids = [w["id"] for w in words]
    texts = [w["text"] for w in words]
    if len(set(ids)) != len(ids) or len(set(texts)) != len(texts):
        raise ValueError("music presentation words and IDs must be unique")
    for word in words:
        th = word["threshold"]
        if (isinstance(th, bool) or not isinstance(th, (int, float))
                or not math.isfinite(th) or not 0 < th <= 1):
            raise ValueError(f"invalid threshold for {word['text']}")
    for stage in config["stages"]:
        if not stage["answers"] or any(w not in texts for w in stage["answers"]):
            raise ValueError(f"unknown/empty answers in music stage {stage['id']}")
    answers = build_answer_entries(
        [WordRow(w["id"], w["text"], "ko", w["action_group"]) for w in words],
        thresholds={w["text"]: w["threshold"] for w in words}, strict=True,
    )
    # Retain gerund comparisons that keep 들어 distinct from 흔들어 etc.
    # These are comparison-only; only the twelve requested words are answers.
    extra = build_answer_entries([
        r for r in read_words_csv(ROOT / "shared/words.csv")
        if r.action_group and r.text not in texts
    ])
    # Removed object names still need to compete with similar accepted words
    # (e.g. 실로폰 / 실로폰채), without becoming lesson answers themselves.
    extra.extend(build_answer_entries([
        WordRow(f"comparison_only_{i}", a["text"], "ko", a["action_group"])
        for i, a in enumerate(config.get("comparison_only", []))
    ], strict=True))
    # Contexts supplement whole-utterance validation only after a separate,
    # non-overlapping prefix and answer have matched. They are not aliases:
    # an unguarded long alias can mistake a noun alone for a short verb.
    contexts = config.get("sentence_prefixes", {})
    if not isinstance(contexts, dict) or any(w not in texts for w in contexts):
        raise ValueError("sentence_prefixes must map known answer words to prefix lists")
    g2p = get_g2p("ko")
    for answer, word in zip(answers, words):
        prefixes = contexts.get(word["text"], [])
        if (not isinstance(prefixes, list)
                or any(not isinstance(p, str) or not p.strip() for p in prefixes)):
            raise ValueError("sentence_prefixes entries must be non-empty strings")
        if prefixes:
            answer["context_prefixes"] = [
                {"text": prefix.strip(), "phonemes": g2p.to_ipa(prefix.strip())}
                for prefix in dict.fromkeys(prefixes)
            ]
    add_contrastive_alternatives(answers + extra)
    # Whole-utterance validation must allow polite endings, particularly
    # for a one-syllable word where 요 doubles the sequence length. These
    # aliases support only their own group, not another action's anchor.
    polite = build_answer_entries([
        WordRow(f"comparison_{i}", a["text"], "ko", a["action_group"])
        for i, a in enumerate(config.get("comparison_aliases", []))
    ])
    for answer, word in zip(answers, words):
        answer["contrastive_alternatives"].extend(
            {k: a[k] for k in ("text", "phonemes", "action_group")}
            for a in polite if a["action_group"] == answer["action_group"]
        )
        for key in ("confirmation_frames", "min_whole_score", "contrast_margin", "short_utterance_policy", "acoustic_keyword_policy"):
            if key in word:
                answer[key] = word[key]
    validate_catalog(answers)
    return {
        "version": "1.0", "profile_id": config["profile_id"],
        "confusion_matrix_id": config["confusion_matrix_id"],
        "presentation_stages": config["stages"],
        "answers": answers,
    }


def main() -> int:
    catalog = build_catalog(json.loads(CONFIG.read_text(encoding="utf-8")))
    content = json.dumps(catalog, ensure_ascii=False, indent=2) + "\n"
    for path in DESTINATIONS:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    # The Unity presentation API reads its own package resources, so updating
    # the package also updates this policy without replacing project JSONs.
    for file in ("ko_child_v2.json", "wav2vec2_ko_vocab.json"):
        (BUNDLED_DATA / file).write_bytes((PACKAGE / "Samples~/KoreanData" / file).read_bytes())
    FEEDBACK_DATA.mkdir(parents=True, exist_ok=True)
    (FEEDBACK_DATA / "reference_errors_child.json").write_bytes(
        (ROOT / "shared/reference/reference_errors.json").read_bytes())
    print(f"Music demo: {len(catalog['answers'])} words, {len(catalog['presentation_stages'])} stages; synced {len(DESTINATIONS)} catalogs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
