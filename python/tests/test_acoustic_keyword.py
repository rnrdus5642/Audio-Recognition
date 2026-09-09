import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from python.runtime.matching import ConfusionMatrix, Matcher, StreamingMatcher
from python.runtime.matching.acoustic import AcousticEvidence, validate_acoustic_policy
from python.runtime.recognizer.ko.ctc_evidence import sequence_log_probability

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("target", [[], [0], [0, 0], [0, 1], [1, 0, 1]])
def test_ctc_forward_sum_matches_exhaustive_alignment_enumeration(target):
    probabilities = np.array([[.5,.2,.3],[.1,.4,.5],[.3,.4,.3],[.2,.3,.5]])
    total = 0.
    for alignment in itertools.product(range(3), repeat=4):
        collapsed = [v for i,v in enumerate(alignment) if v != 2 and (i == 0 or v != alignment[i-1])]
        if collapsed == target:
            total += np.prod([probabilities[i,v] for i,v in enumerate(alignment)])
    assert sequence_log_probability(np.log(probabilities), target, 2) == pytest.approx(np.log(total))


@pytest.fixture
def configured():
    catalog = json.loads((ROOT/'shared/targets_music_demo.json').read_text(encoding='utf-8'))
    a = next(a for a in catalog['answers'] if a['text'] == '잡아')
    m = Matcher.for_streaming(ConfusionMatrix.from_json(ROOT/'shared/confusion_matrices/ko_child_v2.json'))
    return a,m


def test_same_raw_phonemes_require_acoustic_support_and_two_fresh_frames(configured):
    a,m = configured
    ipa = ['tɕ','a','k','a']  # Both 잡아 and 작아 recordings produced this sequence.
    support = AcousticEvidence('잡아','작아',-4.,-1.,-1.)
    wrong = AcousticEvidence('잡아','작아',-14.,-.2,-.2)
    assert not m.best_match(ipa,[a]).passed
    assert not m.best_match(ipa,[a],{'잡아':wrong}).passed
    sm = StreamingMatcher(m,[a])
    assert sm.push(ipa,{'잡아':support}) is None
    assert sm.push(ipa) is None  # Missing current-frame evidence breaks the streak.
    assert sm.push(ipa,{'잡아':support}) is None
    hit = sm.push(ipa,{'잡아':support})
    assert hit is not None and hit.streak == 2
    assert hit.result.user_phonemes == ipa
    assert hit.result.score < .97 and hit.result.acoustic_evidence == support
    assert hit.feedback.korean_feedback.items == []
    assert hit.feedback.korean_feedback.summary == '발음에 문제 없어요.'
    assert all(w.reason == 'acoustic_keyword_recovery' for w in hit.feedback.feedback_review.withheld)
    assert hit.feedback.raw_korean_feedback.items


def test_acoustic_rescue_is_opt_in_short_and_still_honors_competitor(configured):
    a,m = configured
    ipa = ['tɕ','a','k','a']
    e = {'잡아':AcousticEvidence('잡아','작아',-4.,-1.,-1.)}
    legacy=copy.deepcopy(a);legacy.pop('acoustic_keyword_policy')
    assert not m.best_match(ipa,[legacy],e).passed
    assert not m.best_match(ipa*3,[a],e).passed
    assert not m.best_match([], [a],e).passed
    rival=copy.deepcopy(a);rival['contrast_margin']=0
    rival['contrastive_alternatives'].append(dict(text='작아',phonemes=ipa,action_group='small'))
    assert m.best_match(ipa,[rival],e).competitor_rejected
    assert not m.best_match(ipa,[rival],e).passed
    strict=copy.deepcopy(a);strict['min_whole_score']=.95
    assert not m.best_match(ipa,[strict],e).passed


@pytest.mark.parametrize('value',[float('nan'),float('inf'),float('-inf'),1.])
def test_invalid_model_evidence_never_grants_pass(value,configured):
    a,m=configured
    e=AcousticEvidence('잡아','작아',value,-1.,-1.)
    assert not m.best_match(['tɕ','a','k','a'],[a],{'잡아':e}).passed


@pytest.mark.parametrize('patch',[{'min_score':True},{'min_score':0},{'min_log_probability':0},
    {'max_log_gap':float('nan')},{'max_contrast_log_gap':-1},{'max_extra_phonemes':2.0},
    {'contrast_text':'잡아'},{'contrast_text':''}])
def test_bad_acoustic_policy_is_rejected(patch,configured):
    a,_=configured;a=copy.deepcopy(a);a['acoustic_keyword_policy'].update(patch)
    with pytest.raises(ValueError,match='acoustic_keyword_policy'):validate_acoustic_policy(a)
