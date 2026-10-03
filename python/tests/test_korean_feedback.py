"""Human explanations are part of the shared Python/Unity contract."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from python.build.g2p.ko.jamo_ipa import hangul_to_ipa_phonemes
from python.runtime.matching import MatchResult, PronunciationFeedback
from python.tools._web_feedback import pronunciation_feedback_html


VECTORS = json.loads((Path(__file__).resolve().parents[2] / (
    "unity/Packages/com.domicube.phoneme-matching/Tests/Runtime/korean_feedback_vectors.json"
)).read_text(encoding="utf-8"))


def from_vector(case):
    user = [step[0] for step in case["alignment"] if step[2] != "del"]
    return PronunciationFeedback.from_match(MatchResult(
        score=0.9, distance=0.5, threshold=0.65, passed=True,
        target_id="test", target_text=case["target_text"],
        target_phonemes=case["target_phonemes"], user_phonemes=user,
        window_end=len(user), alignment=case["alignment"],
    ), 2, 2)


@pytest.mark.parametrize("case", VECTORS, ids=[case["name"] for case in VECTORS])
def test_shared_korean_feedback_vectors(case):
    feedback = from_vector(case)
    assert asdict(feedback.korean_feedback) == case["expected"]
    # Every scored difference is explained exactly once, even when grouped.
    explained = [index for item in feedback.korean_feedback.items for index in item.alignment_indices]
    assert sorted(explained) == [
        i for i, step in enumerate(case["alignment"]) if step[2] != "match"
    ]
    assert feedback.schema_version == 2


def test_all_hangul_syllables_reuse_the_canonical_mapping():
    for code in range(0xAC00, 0xD7A4):
        text = chr(code)
        tokens = hangul_to_ipa_phonemes(text)
        case = {"target_text": text, "target_phonemes": tokens,
                "alignment": [(token, token, "match") for token in tokens]}
        guide = from_vector(case).korean_feedback
        assert guide.syllable_mapping_available, text
        assert guide.items == [], text


def test_html_leads_with_korean_and_collapses_ipa_details():
    feedback = from_vector(VECTORS[0])
    html = pronunciation_feedback_html(feedback)
    primary, details = html.split("<details", 1)
    assert "사 → 타" in primary
    assert "‘사과’의 ‘사’가 ‘타’에 가까운 소리로 인식됐어요." in primary
    assert "tʰ" not in primary
    assert "tʰ" in details
    assert "IPA 상세 비교" in details
    assert " open" not in details.split(">", 1)[0]


def test_html_escapes_human_explanations_too():
    feedback = from_vector(VECTORS[0])
    feedback.korean_feedback.summary = "<svg/onload=alert(1)>"
    feedback.korean_feedback.items[0].message = "<img src=x onerror=alert(1)>"
    feedback.korean_feedback.items[0].heard_syllable = "<script>"
    html = pronunciation_feedback_html(feedback)
    assert "<svg" not in html
    assert "<img" not in html
    assert "<script>" not in html
    assert "&lt;svg" in html
    assert "&lt;img" in html
    assert "&lt;script&gt;" in html


def test_serialized_hints_are_detached_from_the_confirmation():
    feedback = from_vector(VECTORS[0])
    message = feedback.korean_feedback.items[0].message
    payload = feedback.to_dict()
    payload["korean_feedback"]["items"][0]["message"] = "changed"
    payload["korean_feedback"]["items"][0]["alignment_indices"].clear()
    assert feedback.korean_feedback.items[0].message == message
    assert feedback.korean_feedback.items[0].alignment_indices == [0]
