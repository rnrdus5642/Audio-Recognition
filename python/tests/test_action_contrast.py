"""Telling one music action from another, without changing what scores.

The lesson asks for one action in two forms - 들어 / 들기 - and five
actions compete. 들기's phonemes are the tail of 흔들기's, so substring
scoring, which may start its window anywhere, reads a child who said
흔들기 as a perfect 들기: 0.997 against a 0.875 pass mark. Scoring those
words whole instead swaps that failure for its mirror image (들기 read as
흔들기), so the catalog ships the OTHER actions alongside each grouped
answer and the runtime uses them for one thing only - refusing a pass it
cannot justify. Never to rank, never to confirm, never to rewrite a score.
"""

import json
from pathlib import Path

import pytest

from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher
from python.runtime.matching.confirmation import PronunciationFeedback
from python.runtime.matching.matcher import OTHER_ACTION_SCORED_HIGHER
from python.runtime.matching.matcher import validate_catalog
from python.build.build_targets import WordRow, build_answer_entries, \
    read_words_csv
from python.tools.test_real_audio import AudioTester

ROOT = Path(__file__).resolve().parents[2]
DATA = (ROOT / "unity/Packages/com.domicube.phoneme-matching"
        / "Samples~/KoreanData")

#: The five actions, each as the two expressions the lesson accepts.
PAIRS = [("들어", "들기"), ("쳐", "치기"), ("놓아", "놓기"),
         ("흔들어", "흔들기"), ("두드려", "두드리기")]


@pytest.fixture(scope="module")
def matrix():
    return ConfusionMatrix.from_json(DATA / "ko_child_v2.json")


@pytest.fixture(scope="module")
def catalog():
    return {a["text"]: a for a in json.loads(
        (DATA / "targets_child.json").read_text(encoding="utf-8"))["answers"]}


@pytest.fixture
def matcher(matrix):
    return Matcher.for_streaming(matrix)


def test_the_known_collision_is_refused_and_says_which_action(
        matcher, catalog):
    """흔들기 spoken while the lift pair is on screen.

    The scored answer keeps the score it earned - the veto is not a
    rescoring - and the result carries the action that fitted better, so
    a screen can say "that sounded like 흔들기" instead of a bare failure.
    """
    result = matcher.best_match(
        catalog["흔들기"]["phonemes"], [catalog["들어"], catalog["들기"]])

    assert result.target_text == "들기"
    assert result.score == pytest.approx(0.997)
    assert result.score >= catalog["들기"]["threshold"]
    assert result.passed is False
    assert result.competitor_rejected is True
    assert result.competitor_reason == OTHER_ACTION_SCORED_HIGHER
    assert result.competitor_text == "흔들기"
    assert result.competitor_score == pytest.approx(1.0)
    assert result.accepted_whole_score == pytest.approx(0.64)


def test_the_reverse_collision_is_still_refused_by_the_pass_mark(
        matcher, catalog):
    """The mirror case, which whole-sequence THRESHOLDS would have let in.

    Nothing here is the veto's doing: 들기 simply does not score like
    흔들기, and the ordinary threshold says so.
    """
    result = matcher.best_match(
        catalog["들기"]["phonemes"], [catalog["흔들어"], catalog["흔들기"]])

    assert result.score == pytest.approx(0.64453125)
    assert result.passed is False
    assert result.competitor_rejected is False


def test_an_unrequested_alias_of_the_winners_own_action_defends_it(
        matcher, catalog):
    """The lesson asked for 두드려; the child said 두드리기 with a stray sound.

    Both are the same instruction, but only one was asked for. Measured
    against the requested form alone the winner is worth 0.671, which
    a different action (흔들기, 0.713) beats - so the answer would be
    taken away for saying the drum action a second legitimate way. The
    alias the build embedded is worth 0.925, and that is what the action
    is really worth here, so nothing is refused. The alias still does not
    become the answer: 두드려 was asked for and 두드려 is returned.
    """
    spoken = ["h"] + catalog["두드리기"]["phonemes"]
    asked = [catalog["두드려"]]

    result = matcher.best_match(spoken, asked)

    assert result.score == pytest.approx(0.8692857142857142)
    assert result.score >= catalog["두드려"]["threshold"]
    assert matcher.whole_score(spoken, catalog["두드려"]["phonemes"]) == \
        pytest.approx(0.6714285714285715)
    assert matcher.whole_score(spoken, catalog["흔들기"]["phonemes"]) == \
        pytest.approx(0.7125)
    assert matcher.whole_score(spoken, catalog["두드리기"]["phonemes"]) == \
        pytest.approx(0.925)
    assert result.target_text == "두드려"
    assert result.passed is True
    assert result.competitor_rejected is False


def test_without_the_embedded_alias_that_case_would_be_refused(
        matcher, catalog):
    """The same frame, with the own-group aliases taken out of the data.

    Characterises what the anchor is for: this is the shape the guard had
    before the aliases were embedded, and it is a wrong answer.
    """
    spoken = ["h"] + catalog["두드리기"]["phonemes"]
    stripped = dict(catalog["두드려"], contrastive_alternatives=[
        r for r in catalog["두드려"]["contrastive_alternatives"]
        if r["action_group"] != "drum"])

    result = matcher.best_match(spoken, [stripped])

    assert result.passed is False
    assert result.competitor_text == "흔들기"


@pytest.mark.parametrize("pair", PAIRS)
def test_every_action_still_confirms_itself(matcher, catalog, pair):
    for spoken in pair:
        answers = [catalog[pair[0]], catalog[pair[1]]]
        result = matcher.best_match(catalog[spoken]["phonemes"], answers)
        assert result.target_text == spoken
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert result.competitor_rejected is False


def test_an_action_the_question_also_accepts_is_not_a_competitor(
        matcher, catalog):
    """Both actions on screen: hearing either one is a right answer.

    The rule is read off the catalog - the rival's group is represented
    among the answers - so no runtime code has to know that 흔들기 and
    들기 are different instructions.
    """
    answers = [catalog["들어"], catalog["들기"],
               catalog["흔들어"], catalog["흔들기"]]

    result = matcher.best_match(catalog["흔들기"]["phonemes"], answers)

    assert result.target_text == "흔들기"
    assert result.passed is True
    assert result.competitor_rejected is False


def test_an_alias_with_the_same_sounds_is_not_a_competitor(matcher, catalog):
    """A rival the question accepts under a different name.

    Same phonemes as an accepted answer means the child saying it was
    right, whatever the catalog calls it, so it cannot be the reason for
    a refusal. Built here rather than found in the shipped data: no two
    shipped actions share a pronunciation, and this must hold if two ever
    do.
    """
    borrowed = dict(catalog["흔들기"], id="borrowed", action_group="borrowed")
    answers = [catalog["들어"], catalog["들기"], borrowed]

    result = matcher.best_match(catalog["흔들기"]["phonemes"], answers)

    assert result.target_text == "흔들기"
    assert result.passed is True
    assert result.competitor_rejected is False


def rival(phonemes, text="흔들기", group="shake"):
    return {"text": text, "phonemes": phonemes, "action_group": group}


def accepted(catalog, phonemes, rivals, threshold=0.9):
    """One passing answer of a made-up action, with rivals of another."""
    return dict(catalog["들기"], phonemes=phonemes, threshold=threshold,
                action_group="lift", contrastive_alternatives=rivals)


def test_a_tie_does_not_veto(matcher, catalog):
    """Equal is not better, and only better may refuse.

    Two phonemes the matrix has never seen cost the same default
    insertion, so answer and rival are worth exactly the same against
    this speech - which is no evidence that the other action was meant.
    The same setup with a rival that fits strictly better does refuse,
    so this is the boundary and not an inert case.
    """
    spoken = ["qq", "rr"]
    tied = matcher.best_match(
        spoken, [accepted(catalog, ["qq"], [rival(["rr"])])])
    assert matcher.whole_score(spoken, ["qq"]) == matcher.whole_score(
        spoken, ["rr"])
    assert tied.passed is True
    assert tied.competitor_rejected is False

    beaten = matcher.best_match(
        spoken, [accepted(catalog, ["qq"], [rival(["qq", "rr"])])])
    assert beaten.passed is False
    assert beaten.competitor_rejected is True


def test_the_veto_never_reranks(matcher, catalog):
    """A refusal is a refusal, not a different answer.

    Winner, score, window and alignment are whatever the ordinary
    decision made them; only `passed` moves, and only downwards.
    """
    spoken = catalog["흔들기"]["phonemes"]
    lift = [catalog["들어"], catalog["들기"]]

    vetoed = matcher.best_match(spoken, lift)
    baseline = matcher.best_match(spoken, [dict(
        catalog["들기"], contrastive_alternatives=None)])

    assert vetoed.target_id == catalog["들기"]["id"]
    assert (vetoed.score, vetoed.distance) == (baseline.score, baseline.distance)
    assert (vetoed.window_start, vetoed.window_end) == (
        baseline.window_start, baseline.window_end)
    assert vetoed.alignment == baseline.alignment
    assert baseline.passed is True and vetoed.passed is False


def test_an_answer_that_ships_no_contrasts_is_untouched(matcher, catalog):
    """Legacy data, and the 25 words that were never in a contrast set."""
    plain = [a for a in catalog.values() if "contrastive_alternatives" not in a]
    assert len(plain) == 25

    result = matcher.best_match(catalog["사과"]["phonemes"], [catalog["사과"]])

    assert result.passed is True
    assert result.competitor_rejected is False
    assert result.competitor_text is None


@pytest.mark.parametrize("broken", [
    "흔들기",
    {"text": "흔들기", "phonemes": ["h"]},
    {"text": "흔들기", "action_group": "shake"},
    {"text": "", "phonemes": ["h"], "action_group": "shake"},
    {"text": "흔들기", "phonemes": [], "action_group": "shake"},
    {"text": "흔들기", "phonemes": ["h", 3], "action_group": "shake"},
])
def test_malformed_contrast_metadata_is_refused_not_scored(
        matcher, catalog, broken):
    """A build that went wrong must not read as "nothing to check".

    Absent alternatives are legacy data and mean exactly that; present
    but unreadable ones would silently drop the protection the field was
    added for. The C# loader refuses the same shapes.
    """
    winner = dict(catalog["들기"], contrastive_alternatives=[broken])

    with pytest.raises(ValueError, match="contrastive alternative"):
        matcher.best_match(catalog["흔들기"]["phonemes"], [winner])


def test_contrasts_without_an_action_group_are_refused(matcher, catalog):
    winner = dict(catalog["들기"])
    winner.pop("action_group")

    with pytest.raises(ValueError, match="action_group"):
        matcher.best_match(catalog["흔들기"]["phonemes"], [winner])


@pytest.mark.parametrize("container", [False, 0, "", {}, "들기", 3.0])
def test_a_container_that_is_not_a_list_is_refused_not_ignored(
        matcher, catalog, container):
    """The failure that matters most: protection configured, then dropped.

    `if not alternatives` would read every one of these as "no contrast
    set" and score the frame with the guard silently off, while the C#
    loader refuses the same file. Neither half may decide this alone.
    """
    winner = dict(catalog["들기"], contrastive_alternatives=container)

    with pytest.raises(ValueError, match="not a list"):
        matcher.best_match(catalog["흔들기"]["phonemes"], [winner])


def test_an_empty_list_is_allowed_and_vetoes_nothing(matcher, catalog):
    """A build that found nothing to embed is not a broken build."""
    winner = dict(catalog["들기"], contrastive_alternatives=[])

    result = matcher.best_match(catalog["흔들기"]["phonemes"], [winner])

    assert result.passed is True
    assert result.competitor_rejected is False


@pytest.mark.parametrize("group", ["", " ", "put down", "lift\t", 7, True])
def test_a_group_that_is_not_a_name_is_refused_on_the_answer(
        matcher, catalog, group):
    """Groups are compared for equality, so ' lift' protects nothing."""
    winner = dict(catalog["들기"], action_group=group)

    with pytest.raises(ValueError, match="action_group"):
        matcher.best_match(catalog["흔들기"]["phonemes"], [winner])


@pytest.mark.parametrize("group", ["", " lift", 7, None])
def test_a_group_that_is_not_a_name_is_refused_on_an_alternative(
        matcher, catalog, group):
    winner = dict(catalog["들기"], contrastive_alternatives=[
        {"text": "흔들기", "phonemes": ["h"], "action_group": group}])

    with pytest.raises(ValueError, match="action_group"):
        matcher.best_match(catalog["흔들기"]["phonemes"], [winner])


def test_a_candidates_own_group_is_validated_too(matcher, catalog):
    """Not just the winner's: any accepted answer can exclude a rival."""
    other = dict(catalog["흔들어"], action_group="sha ke")

    with pytest.raises(ValueError, match="action_group"):
        matcher.best_match(catalog["들기"]["phonemes"],
                           [catalog["들기"], other])


def test_the_shipped_catalogs_pass_the_load_time_check():
    """Every copy, read the way the loaders read them."""
    for name in ("shared/targets_child.json",
                 "unity/Assets/StreamingAssets/targets_child.json",
                 "unity/Packages/com.domicube.phoneme-matching/"
                 "Samples~/KoreanData/targets_child.json"):
        answers = json.loads(
            (ROOT / name).read_text(encoding="utf-8"))["answers"]
        assert len(validate_catalog(answers)) == 35


def test_the_load_time_check_names_the_answer_that_is_wrong():
    with pytest.raises(ValueError, match="'들기'"):
        validate_catalog([{"text": "들기", "contrastive_alternatives": 0}])


# --------------------------------------------------------------------------
# What the web tool refuses to load
# --------------------------------------------------------------------------


MATRIX = ROOT / "shared/confusion_matrices/ko_child_v2.json"


def catalog_file(tmp_path, *answers):
    """A synthetic public catalog, written the way the build writes one."""
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(
        {"version": "2.0.0", "confusion_matrix_id": "ko_child_v2",
         "answers": list(answers)}, ensure_ascii=False), encoding="utf-8")
    return path


def row(**overrides):
    answer = {"id": "lift_gerund", "text": "들기", "language": "ko",
              "phonemes": ["t", "ɯ", "l", "k", "i"], "threshold": 0.875}
    answer.update(overrides)
    return answer


@pytest.fixture
def no_model(monkeypatch):
    """Loading a catalog must not reach for an acoustic model."""
    def boom(self, language):
        raise AssertionError("the constructor loaded a recogniser")

    monkeypatch.setattr(AudioTester, "_recognizer_for", boom)


@pytest.mark.parametrize("broken", [
    row(action_group="lift", contrastive_alternatives=0),
    row(action_group="lift", contrastive_alternatives={}),
    row(action_group="lift", contrastive_alternatives=""),
    row(action_group="lift", contrastive_alternatives=[
        {"text": "흔들기", "phonemes": [], "action_group": "shake"}]),
    row(action_group="lift", contrastive_alternatives=[
        {"text": "흔들기", "phonemes": ["h", ""], "action_group": "shake"}]),
    row(action_group="lift", contrastive_alternatives=[
        {"text": "흔들기", "phonemes": ["h"], "action_group": " shake"}]),
    row(action_group="li ft"),
    row(action_group=""),
    row(contrastive_alternatives=[
        {"text": "흔들기", "phonemes": ["h"], "action_group": "shake"}]),
])
def test_the_tool_refuses_a_broken_catalog_when_it_loads_it(
        tmp_path, no_model, broken):
    """Not only for the row that wins a frame.

    The broken row here is never asked for by any test that follows: a
    catalog is refused for what it says, not for what happens to be
    scored, or a lesson that never uses 들기 would ship the fault to the
    device unnoticed.
    """
    path = catalog_file(tmp_path, row(id="apple", text="사과",
                                      phonemes=["s", "a", "k", "w", "a"]),
                        broken)

    with pytest.raises(ValueError, match="action_group|list|phonemes"):
        AudioTester(targets_path=path, matrix_path=MATRIX)


def test_a_legacy_catalog_without_any_contrast_metadata_still_loads(
        tmp_path, no_model):
    path = catalog_file(tmp_path, row(), row(id="apple", text="사과",
                                             phonemes=["s", "a"]))

    tester = AudioTester(targets_path=path, matrix_path=MATRIX)

    assert len(tester.targets["answers"]) == 2
    assert tester._answer_index["들기"][0] == "lift_gerund"


def test_a_grouped_catalog_with_an_empty_list_loads(tmp_path, no_model):
    """An empty list is a build that found nothing to embed."""
    path = catalog_file(tmp_path,
                        row(action_group="lift", contrastive_alternatives=[]))

    tester = AudioTester(targets_path=path, matrix_path=MATRIX)

    assert tester.targets["answers"][0]["contrastive_alternatives"] == []


def test_the_live_only_tool_still_starts_without_a_catalog(no_model):
    tester = AudioTester(targets_path=None, matrix_path=MATRIX)

    assert tester.targets == {"answers": []}
    assert tester._answer_index == {}


def test_the_shipped_catalog_loads_through_the_constructor(no_model):
    tester = AudioTester(targets_path=ROOT / "shared/targets_child.json",
                         matrix_path=MATRIX)

    assert len(tester.targets["answers"]) == 35


def test_a_vetoed_frame_neither_streaks_nor_confirms_on_one_frame(
        matcher, catalog):
    """The veto lands before the streak and before the escape.

    A refused frame is a frame nobody won, so the streak resets and the
    one-exact-frame shortcut - which only ever applies to a passing
    answer - cannot fire either.
    """
    lift = [catalog["들어"], catalog["들기"]]
    sm = StreamingMatcher(matcher, lift)

    assert sm.push(catalog["흔들기"]["phonemes"]) is None
    assert sm.streak == 0
    assert sm.push(catalog["흔들기"]["phonemes"]) is None
    assert sm.streak == 0

    # The same two frames of the action that WAS asked for still confirm.
    sm.reset()
    assert sm.push(catalog["들기"]["phonemes"]) is None
    hit = sm.push(catalog["들기"]["phonemes"])
    assert hit is not None and hit.result.target_text == "들기"


def test_the_confirmation_payload_is_unchanged(matcher, catalog):
    """Feedback describes the utterance, and the veto adds nothing to it.

    A confirmed answer is one the veto did not touch, so its payload is
    the same one `PronunciationFeedback.from_match` always built.
    """
    spoken = catalog["들기"]["phonemes"]
    result = matcher.best_match(spoken, [catalog["들어"], catalog["들기"]])

    payload = PronunciationFeedback.from_match(result, frames=2, streak=2)

    assert payload.passed is True
    assert payload.to_dict() == PronunciationFeedback.from_match(
        matcher.best_match(spoken, [catalog["들기"]]), 2, 2).to_dict()
    assert "competitor_rejected" not in payload.to_dict()


def licensed(matrix, catalog):
    """A word licensed to confirm on one exact frame, and its matcher.

    The escape needs the matcher to match the snapshot calibration wrote,
    so both are built together here. The narrow context limit is what
    makes a 흔들기 frame score exactly 1.0 against 들기: the slice that
    gets scored is the last five phonemes, which are 들기's, while the
    whole utterance is still 흔들기's.
    """
    profile = matrix.streaming_profile
    scorer = Matcher(matrix, skip_cost=profile["skip_cost"],
                     coverage=profile["coverage"], context_mult=0.4)
    answer = dict(catalog["들기"], allow_exact_single_frame=True)
    answer["exact_single_frame_calibration"] = {
        "matrix_id": matrix.matrix_id,
        "streaming_profile": {"skip_cost": scorer.skip_cost,
                              "coverage": scorer.coverage,
                              "context_mult": 0.4},
        "consecutive": 2,
        "text": answer["text"],
        "phonemes": answer["phonemes"],
        "threshold": answer["threshold"],
    }
    return scorer, answer


def test_the_veto_precedes_the_one_exact_frame_shortcut(matrix, catalog):
    """A licensed 1.0 frame is still refused when it is the wrong action.

    Without the contrast set this frame confirms on its own, streak 1 -
    that is the calibrated shortcut working. The guard runs before both
    confirmation branches, so the same frame confirms nothing and leaves
    no streak behind for the next one to build on.
    """
    scorer, answer = licensed(matrix, catalog)
    spoken = catalog["흔들기"]["phonemes"]
    unguarded = dict(answer)
    unguarded.pop("contrastive_alternatives")

    shortcut = StreamingMatcher(scorer, [unguarded], consecutive=2).push(spoken)
    assert shortcut is not None and shortcut.streak == 1
    assert shortcut.result.score == 1.0

    guarded = StreamingMatcher(scorer, [answer], consecutive=2)
    assert guarded.push(spoken) is None
    assert guarded.streak == 0

    refused = scorer.best_match(spoken, [answer])
    assert refused.score == 1.0, "the score it earned is left alone"
    assert refused.passed is False
    assert refused.competitor_text == "흔들기"


# --------------------------------------------------------------------------
# What the build writes
# --------------------------------------------------------------------------


def test_the_builder_embeds_every_grouped_answer_including_own_aliases():
    """The rule lives in the catalog, so the runtime knows no Korean.

    Own-group aliases are embedded too, because the anchor needs them: a
    question may ask for one form of an action while the child says the
    other. Splitting them from the rivals is the runtime's job, and it
    does it by comparing group names.
    """
    entries = build_answer_entries([
        WordRow("lift_plain", "들어", "ko", "lift"),
        WordRow("lift_gerund", "들기", "ko", "lift"),
        WordRow("hit_plain", "쳐", "ko", "strike"),
        WordRow("apple", "사과", "ko"),
    ])
    by_text = {e["text"]: e for e in entries}

    assert [r["text"] for r in by_text["들어"]["contrastive_alternatives"]] \
        == ["들어", "들기", "쳐"]
    assert [r["text"] for r in by_text["쳐"]["contrastive_alternatives"]] \
        == ["들어", "들기", "쳐"]
    rival = by_text["쳐"]["contrastive_alternatives"][0]
    assert rival == {"text": "들어", "phonemes": by_text["들어"]["phonemes"],
                     "action_group": "lift"}
    assert "contrastive_alternatives" not in by_text["사과"]
    assert "action_group" not in by_text["사과"]


def test_one_group_on_its_own_can_anchor_but_has_no_rival():
    """A single-group catalog embeds itself and vetoes nothing."""
    entries = build_answer_entries([
        WordRow("lift_plain", "들어", "ko", "lift"),
        WordRow("lift_gerund", "들기", "ko", "lift"),
    ])

    assert all(e["action_group"] == "lift" for e in entries)
    assert all([r["action_group"] for r in e["contrastive_alternatives"]]
               == ["lift", "lift"] for e in entries)

    matrix = ConfusionMatrix.from_json(DATA / "ko_child_v2.json")
    result = Matcher.for_streaming(matrix).best_match(
        entries[0]["phonemes"], [dict(entries[0], threshold=0.5)])
    assert result.passed is True
    assert result.competitor_rejected is False


def test_a_group_that_is_not_a_name_is_refused(tmp_path):
    path = tmp_path / "words.csv"
    path.write_text("answer_id,text,language,action_group\n"
                    "lift_plain,들어,ko,li ft\n", encoding="utf-8-sig")

    with pytest.raises(ValueError, match="action_group"):
        read_words_csv(path)


def test_a_file_without_the_column_reads_as_it_always_did(tmp_path):
    path = tmp_path / "words.csv"
    path.write_text("answer_id,text,language\nmom,엄마,ko\n",
                    encoding="utf-8-sig")

    assert read_words_csv(path) == [WordRow("mom", "엄마", "ko", "")]


def test_the_shipped_catalog_groups_the_ten_music_words(catalog):
    grouped = {t: a for t, a in catalog.items() if a.get("action_group")}

    assert sorted(grouped) == sorted(w for pair in PAIRS for w in pair)
    assert len({a["action_group"] for a in grouped.values()}) == len(PAIRS)
    assert {a["action_group"] for a in grouped.values()} == {
        "lift", "strike", "put_down", "shake", "drum"}
    for text, answer in grouped.items():
        rivals = answer["contrastive_alternatives"]
        assert len(rivals) == 10, "every grouped answer, own aliases included"
        assert sum(1 for r in rivals
                   if r["action_group"] == answer["action_group"]) == 2
        assert all(r["phonemes"] == catalog[r["text"]]["phonemes"]
                   for r in rivals)


def test_the_live_web_candidate_inherits_the_contrast_set(matcher):
    """Typing the words into the live tab is the same question.

    The web builder hands the catalog row through, ids and all, so a
    candidate typed by hand carries the group and the alternatives - and
    is refused for the same reason the packaged lesson would refuse it.
    """
    tester = AudioTester(targets_path=ROOT / "shared/targets_child.json",
                         matrix_path=ROOT / "shared/confusion_matrices"
                                            "/ko_child_v2.json")
    live = []
    for i, word in enumerate(("들어", "들기")):
        candidate = dict(tester._make_custom_target(word, "ko")[1])
        candidate["id"] = f"live_{i}"   # unique - the streak keys on id
        live.append(candidate)

    assert [c["action_group"] for c in live] == ["lift", "lift"]
    spoken = tester._make_custom_target("흔들기", "ko")[1]["phonemes"]

    result = matcher.best_match(spoken, live)

    assert result.target_id == "live_1"
    assert result.passed is False
    assert result.competitor_text == "흔들기"


def test_the_live_tab_says_why_a_high_score_was_refused(monkeypatch, catalog):
    """The real live callback, not a rendering helper.

    A number over the pass mark next to "not yet" reads as a bug, so the
    frame that the guard refuses has to say what it heard instead. The
    recording also keeps going: nothing was confirmed.
    """
    pytest.importorskip("gradio")
    import numpy as np
    from python.tools.web_test import SAMPLE_RATE, build_app

    tester = AudioTester(
        targets_path=ROOT / "shared/targets_child.json",
        matrix_path=ROOT / "shared/confusion_matrices/ko_child_v2.json")
    spoken = catalog["흔들기"]["phonemes"]

    class Recognizer:
        def recognize(self, audio):
            return []

        def recognize_with_text(self, audio):
            return "흔들기", list(spoken)

    monkeypatch.setattr(tester, "_recognizer_for", lambda language: Recognizer())
    app = build_app(tester)
    try:
        live_step = next(fn.fn for fn in app.fns.values()
                         if fn.fn and fn.fn.__name__ == "live_step")
        chunk = (SAMPLE_RATE, np.zeros(8000, dtype=np.float32))
        state, status = live_step(chunk, {}, "들어, 들기", "ko", 2.5, 2)[:2]
    finally:
        app.close()

    assert not state.get("done"), "a refused frame confirms nothing"
    assert state["sm"].streak == 0
    assert "흔들기" in status
    assert "인정하지 않았습니다" in status
    assert "0.997" in status, "the score it earned is still shown"


def test_the_reason_escapes_what_it_prints():
    """Catalog text reaches the page; it is data, not markup."""
    from python.runtime.matching.matcher import MatchResult
    from python.tools.web_test import _competitor_md

    html = _competitor_md(MatchResult(
        score=1.0, distance=0.0, target_id="a", target_text="들기",
        target_phonemes=["t"], user_phonemes=["t"], passed=False,
        competitor_rejected=True, competitor_text="<b>흔들기</b> & co",
        competitor_score=1.0, accepted_whole_score=0.5))

    assert "&lt;b&gt;흔들기&lt;/b&gt; &amp; co" in html
    assert "<b>흔들기</b>" not in html
