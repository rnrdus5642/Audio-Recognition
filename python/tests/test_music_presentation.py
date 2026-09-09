"""Presentation policy boundaries and real Python/C# catalog wiring.

These use phoneme fixtures to test decisions, not to claim acoustic accuracy.
Actual recording results are reported separately by evaluate_music_demo.
"""
import copy
import json
from pathlib import Path

import pytest

from python.build.build_music_presentation import BUNDLED_DATA, CONFIG, DESTINATIONS, PACKAGE, build_catalog
from python.build.g2p import get_g2p
from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher
from python.runtime.matching.matcher import validate_catalog

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def catalog():
    return build_catalog(json.loads(CONFIG.read_text(encoding="utf-8")))


@pytest.fixture
def matcher():
    return Matcher.for_streaming(ConfusionMatrix.from_json(
        ROOT / "shared/confusion_matrices/ko_child_v2.json"))


def test_generated_unity_catalogs_match_source(catalog):
    for path in DESTINATIONS[1:]:
        assert json.loads(path.read_text(encoding="utf-8")) == catalog
    assert len(catalog["answers"]) == 12
    assert len(catalog["presentation_stages"]) == 9
    for name in ("ko_child_v2.json", "wav2vec2_ko_vocab.json"):
        assert (BUNDLED_DATA / name).read_bytes() == (PACKAGE / "Samples~/KoreanData" / name).read_bytes()


def test_all_words_honor_their_own_frames_even_with_another_session_default(catalog, matcher):
    for answer in catalog["answers"]:
        sm = StreamingMatcher(matcher, [answer], consecutive=5)
        n = answer["confirmation_frames"]
        for _ in range(n - 1):
            assert sm.push(answer["phonemes"]) is None
        hit = sm.push(answer["phonemes"])
        assert hit is not None, answer["text"]
        assert hit.streak == hit.required_frames == n
        sm.reset()
        assert sm.streak == 0


def test_stages_accept_only_their_expressions_on_canonical_phonemes(catalog, matcher):
    by = {a["text"]: a for a in catalog["answers"]}
    for stage in catalog["presentation_stages"]:
        cands = [by[w] for w in stage["answers"]]
        for answer in catalog["answers"]:
            sm = StreamingMatcher(matcher, cands)
            hit = None
            for _ in range(6):
                hit = sm.push(answer["phonemes"])
                if hit:
                    break
            assert bool(hit) == (answer["text"] in stage["answers"]), (stage["id"], answer["text"])


@pytest.mark.parametrize("word", ["실로폰", "탬버린", "북"])
def test_removed_object_names_do_not_confirm_a_similar_remaining_answer(word, catalog, matcher):
    assert word not in {a["text"] for a in catalog["answers"]}
    sounds = get_g2p("ko").to_ipa(word)
    for stage in catalog["presentation_stages"]:
        candidates = [a for a in catalog["answers"] if a["text"] in stage["answers"]]
        sm = StreamingMatcher(matcher, candidates)
        assert all(sm.push(sounds) is None for _ in range(6)), stage["id"]


@pytest.mark.parametrize("word", ["내려와", "내려와요", "내려가", "데려와", "작아", "앉아", "연습해", "잡아먹어"])
def test_new_words_do_not_accept_similar_different_actions(word, catalog, matcher):
    sounds = get_g2p("ko").to_ipa(word)
    for stage in catalog["presentation_stages"]:
        candidates = [a for a in catalog["answers"] if a["text"] in stage["answers"]]
        sm = StreamingMatcher(matcher, candidates)
        assert all(sm.push(sounds) is None for _ in range(6)), stage["id"]


def test_lift_margin_recovers_close_match_but_still_rejects_shaking(catalog, matcher):
    answer = copy.deepcopy(next(a for a in catalog["answers"] if a["text"] == "들어"))
    sounds = get_g2p("ko").to_ipa("그려")
    strict = dict(answer, contrast_margin=0.0)
    assert matcher.best_match(sounds, [strict]).competitor_rejected
    assert matcher.best_match(sounds, [answer]).passed
    wrong = matcher.best_match(get_g2p("ko").to_ipa("흔들어"), [answer])
    assert not wrong.passed


@pytest.mark.parametrize("word", ["들어", "잡아", "쳐", "연주해", "놓아", "놔", "내려놔", "흔들어", "두드려", "때려"])
def test_polite_endings_still_pass_the_whole_utterance_check(word, catalog, matcher):
    answer = next(a for a in catalog["answers"] if a["text"] == word)
    sm = StreamingMatcher(matcher, [answer])
    sounds = get_g2p("ko").to_ipa(word + "요")
    hits = [sm.push(sounds) for _ in range(answer["confirmation_frames"])]
    assert hits[-1] is not None


def test_whole_floor_blocks_fragment_but_does_not_hide_another_passing_answer(matcher):
    user = ["k", "u"] * 9 + ["s", "a", "k", "w", "a"]
    fragment = {"id": "fragment", "text": "사과", "phonemes": user[-5:],
                "threshold": 0.5, "min_whole_score": 0.8}
    rejected = matcher.best_match(user, [fragment])
    assert rejected.score > fragment["threshold"]
    assert rejected.whole_rejected and not rejected.passed
    full = {"id": "full", "text": "전체", "phonemes": user, "threshold": 0.5}
    result = matcher.best_match(user, [fragment, full])
    assert result.passed and result.target_id == "full"


def test_ungrouped_answer_cannot_borrow_another_words_whole_score(matcher):
    a = {"id": "a", "text": "a", "phonemes": ["a"], "threshold": .5,
         "min_whole_score": .9}
    b = {"id": "b", "text": "b", "phonemes": ["k", "a"], "threshold": 1.0}
    assert matcher.accepted_whole_score(["k", "a"], [a, b], a) < .9


def test_phrase_context_requires_separate_prefix_and_verb_evidence(matcher):
    g2p = get_g2p("ko")
    answer = {"id": "hit", "text": "쳐", "phonemes": g2p.to_ipa("쳐"),
              "threshold": .69, "min_whole_score": .65}
    phrase = g2p.to_ipa("실로폰채를 쳐")
    assert matcher.best_match(phrase, [answer]).whole_rejected
    baseline = copy.deepcopy(answer)
    answer["context_prefixes"] = [{"text": "실로폰채를", "phonemes": g2p.to_ipa("실로폰채를")}]
    assert matcher.best_match(phrase, [answer]).passed
    for text in ("실로폰채", "실로폰채를"):
        user = g2p.to_ipa(text)
        assert matcher.accepted_whole_score(user, [answer], answer) == pytest.approx(
            matcher.accepted_whole_score(user, [baseline], baseline)), text
        assert not matcher.best_match(user, [answer]).passed, text
    for text in ("실로폰채를 들어", "실로폰채를 흔들어"):
        assert not matcher.best_match(g2p.to_ipa(text), [answer]).passed, text


@pytest.mark.parametrize("contexts", [False, {}, "phrase", [None], [{}],
    [{"text": "a", "phonemes": []}],
    [{"text": "a", "phonemes": [1]}],
    [{"text": "a", "phonemes": [" "]}]])
def test_invalid_contexts_are_rejected_at_load_and_runtime(contexts, matcher):
    a = {"id": "a", "text": "a", "phonemes": ["a"], "context_prefixes": contexts}
    with pytest.raises(ValueError, match="context_prefixes"):
        validate_catalog([a])
    with pytest.raises(ValueError, match="context_prefixes"):
        matcher.best_match(["a"], [a])


def test_short_policy_recovers_model_substitution_without_relaxing_long_fragments(matcher):
    g2p = get_g2p("ko")
    answer = {"id": "play", "text": "연주해", "phonemes": g2p.to_ipa("연주해"),
              "threshold": .90, "confirmation_frames": 2}
    user = g2p.to_ipa("연심해")  # Observed TTS recognizer output; not a new answer.
    assert not matcher.best_match(user, [answer]).passed
    answer["short_utterance_policy"] = {"threshold": .75, "min_whole_score": .75,
                                         "max_extra_phonemes": 2}
    result = matcher.best_match(user, [answer])
    assert result.passed and result.used_short_utterance_policy
    assert result.threshold == .75
    stream = StreamingMatcher(matcher, [answer])
    assert stream.push(user) is None
    assert stream.push(user) is not None
    assert not matcher.best_match(["k", "u"] * 9 + user, [answer]).passed
    assert not matcher.best_match(["h", "ɛ"], [answer]).passed
    answer["action_group"] = "play"
    answer["contrastive_alternatives"] = [{"text": "rival", "phonemes": user, "action_group": "other"}]
    assert matcher.best_match(user, [answer]).competitor_rejected


def test_configured_object_commands_confirm_in_every_stage_that_accepts_the_verb(catalog, matcher):
    g2p = get_g2p("ko")
    for answer in catalog["answers"]:
        for prefix in answer.get("context_prefixes", []):
            for ending in ("", "요"):
                phrase = prefix["text"] + " " + answer["text"] + ending
                for stage in catalog["presentation_stages"]:
                    if answer["text"] not in stage["answers"]:
                        continue
                    candidates = [a for a in catalog["answers"] if a["text"] in stage["answers"]]
                    stream = StreamingMatcher(matcher, candidates)
                    hits = [stream.push(g2p.to_ipa(phrase)) for _ in range(2)]
                    assert hits[-1] is not None, (stage["id"], phrase)


@pytest.mark.parametrize("text", ["실로폰채", "실로폰채를", "실로폰채를 들어", "실로폰을 흔들어"])
def test_stage_two_does_not_take_object_or_different_action_as_striking(text, catalog, matcher):
    stage = catalog["presentation_stages"][1]
    candidates = [a for a in catalog["answers"] if a["text"] in stage["answers"]]
    stream = StreamingMatcher(matcher, candidates)
    assert all(stream.push(get_g2p("ko").to_ipa(text)) is None for _ in range(4))


@pytest.mark.parametrize("policy", [False, [], {},
    {"threshold": True, "min_whole_score": .7, "max_extra_phonemes": 2},
    {"threshold": .7, "min_whole_score": 0, "max_extra_phonemes": 2},
    {"threshold": .7, "min_whole_score": .7, "max_extra_phonemes": 2.0},
    {"threshold": .7, "min_whole_score": .7, "max_extra_phonemes": 7}])
def test_invalid_short_policy_fails_on_load_and_in_code(policy, matcher):
    a = {"id": "a", "text": "a", "phonemes": ["a"], "short_utterance_policy": policy}
    with pytest.raises(ValueError, match="short_utterance_policy"):
        validate_catalog([a])
    with pytest.raises(ValueError, match="short_utterance_policy"):
        matcher.best_match(["a"], [a])


@pytest.mark.parametrize("key,value", [
    ("confirmation_frames", 0), ("confirmation_frames", 7),
    ("confirmation_frames", True), ("confirmation_frames", 1.0),
    ("confirmation_frames", "2"), ("min_whole_score", None),
    ("min_whole_score", float("nan")), ("min_whole_score", -0.1),
    ("contrast_margin", "0.08"), ("contrast_margin", 1.1),
])
def test_invalid_policy_fails_on_load_and_in_code(key, value, matcher):
    answer = {"id": "a", "text": "a", "phonemes": ["a"], "threshold": .8, key: value}
    with pytest.raises(ValueError, match=key):
        validate_catalog([answer])
    with pytest.raises(ValueError, match=key):
        StreamingMatcher(matcher, [answer])
    with pytest.raises(ValueError, match=key):
        matcher.best_match(["a"], [answer])


def test_policy_three_cannot_borrow_legacy_single_exact_approval(matcher):
    answer = {"id": "a", "text": "a", "phonemes": ["a"], "threshold": 1.,
              "confirmation_frames": 3, "allow_exact_single_frame": True,
              "exact_single_frame_calibration": {
                  "matrix_id": matcher.matrix.matrix_id, "text": "a", "phonemes": ["a"],
                  "threshold": 1., "consecutive": 2,
                  "streaming_profile": matcher.matrix.streaming_profile,
              }}
    sm = StreamingMatcher(matcher, [answer])
    assert sm.push(["a"]) is None
    assert sm.push(["a"]) is None
    assert sm.push(["a"]).streak == 3


def test_music_web_stage_reset_and_file_replay_use_word_policy(catalog, monkeypatch):
    pytest.importorskip("gradio")
    import numpy as np
    from python.tools.test_real_audio import AudioTester
    from python.tools.web_test import build_app
    tester = AudioTester(targets_path=DESTINATIONS[1], matrix_path=
                         ROOT / "shared/confusion_matrices/ko_child_v2.json")
    word = next(a for a in catalog["answers"] if a["text"] == "흔들어")

    class Recognizer:
        def recognize_with_text(self, audio):
            assert len(audio) == 40000
            return word["text"], word["phonemes"]

    monkeypatch.setattr(tester, "_recognizer_for", lambda lang: Recognizer())
    app = build_app(tester)
    try:
        callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
        selected = callbacks["select_music_stage"](8)
        assert selected[0] == "북채\n들어\n잡아"
        assert selected[1] == {}
        assert selected[-1]["recording"] is False
        chunk = (16000, np.zeros(8000, dtype=np.float32))
        live = callbacks["live_step"](chunk, {}, "흔들어", "ko", 2.5, 2)
        assert live[0]["done"] and live[0]["sm"].streak == 1
        assert "정확일치" not in live[1]
        assert "연속 1회" in live[1]
        replay = list(callbacks["replay_file"](chunk, "흔들어", "ko", 2.5, 2))
        assert len(replay) == 1
        assert replay[-1][8] == live[8]
    finally:
        app.close()


def test_music_nwa_confirmation_json_and_screen_withhold_the_same_asr_difference(monkeypatch):
    pytest.importorskip("gradio")
    import numpy as np
    from python.tools.test_real_audio import AudioTester
    from python.tools.web_test import build_app
    tester = AudioTester(targets_path=DESTINATIONS[1], matrix_path=
                         ROOT / "shared/confusion_matrices/ko_child_v2.json")

    class Recognizer:
        def recognize_with_text(self, audio):
            return "와", ["w", "a"]

    monkeypatch.setattr(tester, "_recognizer_for", lambda lang: Recognizer())
    app = build_app(tester)
    try:
        callbacks = {f.fn.__name__: f.fn for f in app.fns.values() if f.fn}
        selected = callbacks["select_music_stage"](3)
        chunk = (16000, np.zeros(8000, dtype=np.float32))
        first = callbacks["live_step"](chunk, {}, selected[0], "ko", 2.5, 2)
        second = callbacks["live_step"](chunk, first[0], selected[0], "ko", 2.5, 2)
        assert second[0]["done"] and second[-1]["recording"] is False
        data = second[8]
        assert data["target_text"] == "놔"
        assert data["feedback_review"]["status"] == "withheld"
        assert data["feedback_review"]["reference_utterances"] == 697
        assert data["korean_feedback"]["items"] == []
        assert data["deletion_count"] == 1
        primary = second[7].split("<details", 1)[0]
        assert "발음에 문제 없어요." in primary
        assert "보류" not in primary
        assert "놔 → 와" not in primary
        assert "빠졌" not in primary
        replay = list(callbacks["replay_file"](chunk, selected[0], "ko", 2.5, 2))
        assert replay[-1][8] == data
    finally:
        app.close()
