import causal_audit as audit
import labels


def ids(text):
    return [labels.VOCAB[t] for t in labels.jamo(text)]


def test_same_letter_keeps_silent_initial_and_nasal_final_separate():
    result = audit.aggregate([{'speaker': 's', 'surface': '앙', 'ids': ids('아')}])
    counts = {(r['position'], r['token']): r for r in result['by_reference_position']}
    assert counts[('initial', 'ㅇ')]['deletions'] == 0
    assert counts[('final', 'ㅇ')]['deletions'] == 1
    assert result['metrics']['errors'] == 1


def test_inserted_jamo_does_not_invent_a_reference_position():
    _, events, _ = audit.align('가', ids('가') + [labels.VOCAB['ㄱ']])
    assert len(events) == 1 and events[0]['op'] == 'insert'
    assert events[0]['position'] is None and events[0]['reference'] is None


def test_raw_prediction_not_normalized_to_canonical_word():
    result = audit.aggregate([{'speaker': 's', 'surface': '쌕', 'ids': ids('책')}])
    assert result['common_errors'] == [{'reference': 'ㅆ', 'hypothesis': 'ㅊ', 'count': 1}]


def test_consonant_vowel_and_boundary_accounting():
    result = audit.aggregate([{'speaker': 's', 'surface': '쿠 레용', 'ids': ids('포레용')}])
    assert result['metrics']['errors'] == 2
    assert result['by_type']['consonant']['substitutions'] == 1
    assert result['by_type']['vowel']['substitutions'] == 1
    assert result['metrics']['reference_jamo'] == len(ids('쿠레용'))
