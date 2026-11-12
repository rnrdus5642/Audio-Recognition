"""Audit the user's 27-stage / 199-entry curriculum against the frozen data.

Keeps the supplied forms and stage memberships intact. Orthographic variants
are counted separately; no production vocabulary, threshold or training edit.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from types import SimpleNamespace
import unicodedata

from evaluate_performance import ROOT, MANIFESTS, BUNDLE, local_rows, safe, write
from word_game import WordGames, StartWordGame


FAMILIES = {
    '들기': '들어 들어요 들어야지 들거야 들자 들거에요 들어야죠 들까? 들까요?',
    '치기': '쳐 쳐요 쳐야지 칠거야 치자 칠거에요 쳐야죠 칠까? 칠까요?',
    '놓기': '놓아 놓아요 놓아야지 놓을거야 놓자 놓을거에요 놓아야죠 놓을까? 놓을까요?',
    '흔들기': '흔들어 흔들어요 흔들어야지 흔들거야 흔들자 흔들거에요 흔들어야죠 흔들까? 흔들까요?',
    '두드리기': '두드려 두드려요 두드려야지 두드릴거야 두드리자 두드릴거에요 두드려야죠 두드릴까? 두드릴까요?',
    '닦기': '닦아 닦아요 닦아야지 닦을거야 닦자 닦을거에요 닦아야죠 닦을까? 닦을까요?',
    '넣기': '넣어 넣어요 넣어야지 넣을거야 넣자 넣을거에요 넣어야죠 넣을까? 넣을까요?',
    '덮기': '덮어 덮어요 덮어야지 덮을거야 덮자 덮을거에요 덮어야죠 덮을까? 덮을까요?',
    '까기': '까 까요 까야지 깔거야 까자 깔거에요 까야죠 깔까? 깔까요?',
    '꺼내기': '꺼내 꺼내요 꺼내야지 꺼낼거야 꺼내자 꺼낼거에요 꺼내야죠 꺼낼까? 꺼낼까요?',
    '주기': '줘 줘요 줘야지 줄거야 주자 줄거에요 줘야죠 줄까? 줄까요?',
    '보내기': '보내 보내요 보내야지 보낼거야 보내자 보낼거에요 보내야죠 보낼까? 보낼까요?',
    '파기': '파 파요 파야지 팔거야 파자 팔거에요 파야죠 팔까? 팔까요?',
    '따기': '따 따요 따야지 딸거야 따자 딸거에요 따야죠 딸까? 딸까요?',
    '담기': '담아 담아요 담아야지 담을거야 담자 담을거에요 담아야죠 담을까? 담을까요?',
}

NOUNS = {'실로폰채','북채','알','먹이','밥','집','화분','흙','씨앗','물','싹'}
PARTICLES = '이 가 을 를 은 는 에 에서 에게 으로 로 와 과 도 만 하고 에는 에도 에서도 으로는 로는'.split()
NOUN_PARTICLES = {noun+particle:noun for noun in NOUNS for particle in PARTICLES}


def noun_particle_hits(text):
    tokens = set(re.sub(r'[^가-힣\s]', ' ', unicodedata.normalize('NFC',text)).split())
    return {NOUN_PARTICLES[token] for token in tokens if token in NOUN_PARTICLES}


def normalize(text):
    return re.sub(r'[?？]+$', '', unicodedata.normalize('NFC', text).strip())


def catalog():
    stages = []
    def add(activity, number, procedure, family, nouns='', omit=''):
        raw = nouns.split() + FAMILIES.get(family, '').split()
        if omit:
            raw.remove(omit)
        stages.append({'id': f'{activity}-{number}', 'activity': activity, 'number': number,
                       'procedure': procedure, 'family': family, 'raw_targets': raw,
                       'targets': [normalize(word) for word in raw]})
    add('music', 1, '실로폰채 들기', '들기', '실로폰채')
    add('music', 2, '실로폰 치기', '치기')
    add('music', 3, '실로폰채 놓기', '놓기')
    add('music', 4, '탬버린 들기', '들기')
    add('music', 5, '탬버린 치기', '치기')
    add('music', 6, '탬버린 흔들기', '흔들기')
    add('music', 7, '탬버린 놓기', '놓기', omit='놓아야죠')
    add('music', 8, '북채 들기', '들기', '북채')
    add('music', 9, '북 두드리기', '두드리기')
    add('dinosaur', 1, '흙속에서 알 꺼내기', '알', '알')
    add('dinosaur', 2, '알 수건으로 닦기', '닦기')
    add('dinosaur', 3, '알을 통에 넣기', '넣기')
    add('dinosaur', 4, '담요로 알을 덮어주기', '덮기')
    add('dinosaur', 5, '알껍질 까기', '까기')
    add('dinosaur', 6, '통에서 아기 공룡 꺼내기', '꺼내기')
    add('dinosaur', 7, '아기 공룡을 테이블에 놓기', '놓기')
    add('dinosaur', 8, '아기 공룡에게 먹이주기', '주기', '먹이 밥')
    add('dinosaur', 9, '공룡 집으로 보내기', '보내기', '집')
    add('plant', 1, '화분 선택하기', '화분', '화분')
    add('plant', 2, '흙 담기', '흙', '흙')
    add('plant', 3, '구멍 파기', '파기')
    add('plant', 4, '씨앗 심기', '씨앗', '씨앗')
    add('plant', 5, '흙 덮기', '덮기')
    add('plant', 6, '물 주기', '물', '물')
    add('plant', 7, '싹트기', '싹', '싹')
    add('plant', 8, '귤 따기', '따기')
    add('plant', 9, '바구니에 담기', '담기')
    expected = {'music': [10,9,9,9,9,9,8,10,9], 'dinosaur': [1,9,9,9,9,9,9,11,10], 'plant': [1,1,9,1,9,1,1,9,9]}
    for activity, counts in expected.items():
        assert [len(s['targets']) for s in stages if s['activity'] == activity] == counts
    return stages


def match_forms(text, trie):
    """Return substring and word-boundary matches, retaining whitespace positions."""
    raw = re.sub(r'[^가-힣\s]', ' ', unicodedata.normalize('NFC', text))
    positions = [i for i, char in enumerate(raw) if not char.isspace()]
    compact = ''.join(raw[i] for i in positions)
    substring, boundary = set(), set()
    for start, char in enumerate(compact):
        node = trie.get(char)
        if node is None:
            continue
        end = start
        while node is not None:
            if '' in node:
                form = node['']
                substring.add(form)
                left, right = positions[start], positions[end]
                if (left == 0 or raw[left-1].isspace()) and (right+1 == len(raw) or raw[right+1].isspace()):
                    boundary.add(form)
            end += 1
            node = node.get(compact[end]) if end < len(compact) else None
    return compact, substring, boundary


def selected_records(pool, limit=12):
    groups = defaultdict(list)
    for row in pool:
        if .1 <= row['seconds'] <= 4.5:
            groups[(row['age'], row['speaker_key'])].append(row)
    rng = random.Random(20261030)
    for rows in groups.values():
        rows.sort(key=lambda r: r['record_id'])
        rng.shuffle(rows)
    ages = [3,4,5,6]
    keys = {age: [key for key in sorted(groups) if key[0] == age] for age in ages}
    for values in keys.values():
        rng.shuffle(values)
    result = []
    while len(result) < limit:
        added = 0
        for age in ages:
            if len(result) >= limit:
                break
            available = [key for key in keys[age] if groups[key]]
            if available:
                key = available[0]
                result.append(groups[key].pop())
                keys[age].remove(key); keys[age].append(key)
                added += 1
        if not added:
            break
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = safe(args.output or ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test' / datetime.now().strftime('curriculum-%Y%m%d-%H%M%S'))
    output.mkdir(parents=True, exist_ok=True)
    stages = catalog()
    words = sorted({word for stage in stages for word in stage['targets']})
    assert len(stages) == 27 and sum(len(s['targets']) for s in stages) == 199 and len(words) == 146
    vocab = json.loads((BUNDLE/'vocab.json').read_text(encoding='utf-8'))
    from sys import path as python_path
    python_path.insert(0, str(Path(__file__).resolve().parents[1] / 'child_orthographic_release'))
    from inference import decode
    tokens = {v:k for k,v in vocab.items()}
    games = WordGames(SimpleNamespace(ready=True), None, None)
    for stage in stages:
        response = games.start(StartWordGame(targets=stage['raw_targets']))
        assert response['settings']['targets'] == stage['targets']
    variants = {word: word.replace('거에요','거예요') for word in words if '거에요' in word}
    forms = sorted(set(words) | set(variants.values()))
    trie = {}
    for form in forms:
        node = trie
        for char in form:
            node = node.setdefault(char,{})
        node[''] = form
    form_stats = {form: {} for form in forms}
    available, context_available = defaultdict(list), defaultdict(list)
    contract = json.loads((ROOT/'04_Models_and_Experiments/orthographic-jamo-20261027/main/contract.json').read_text(encoding='utf-8'))
    scan = {}
    for split in ('train','dev'):
        manifest = MANIFESTS/f'{split}.jsonl.gz'
        actual_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert actual_hash == contract['manifest_hashes'][manifest.name]
        speakers = defaultdict(set)
        for form in forms:
            form_stats[form][split] = {'substring_utterances':0, 'boundary_utterances':0, 'standalone_utterances':0,
                                     'noun_particle_utterances':0,
                                     'boundary_by_age':{str(age):0 for age in (3,4,5,6)}}
        count = 0
        with gzip.open(manifest,'rt',encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                compact, substring, boundary = match_forms(row['text'],trie)
                noun_hits = noun_particle_hits(row['text'])
                for noun in noun_hits:
                    form_stats[noun][split]['noun_particle_utterances'] += 1
                for form in substring:
                    form_stats[form][split]['substring_utterances'] += 1
                for form in boundary:
                    stat = form_stats[form][split]
                    stat['boundary_utterances'] += 1
                    stat['boundary_by_age'][str(row['age'])] += 1
                    speakers[form].add(row['speaker_key'])
                if compact in form_stats:
                    form_stats[compact][split]['standalone_utterances'] += 1
                    if split == 'dev':
                        available[compact].append({**row,'group':'dev_curriculum'})
                elif split == 'dev' and .1 <= row['seconds'] <= 15.0:
                    for form in boundary | noun_hits:
                        context_available[form].append({**row,'group':'dev_context'})
                count += 1
                if count % 100000 == 0:
                    print(json.dumps({'phase':'manifest','split':split,'done':count}),flush=True)
        for form in forms:
            form_stats[form][split]['boundary_speakers'] = len(speakers[form])
        scan[split] = {'rows':count,'sha256':actual_hash}
    local = local_rows()
    word_results, selected, selected_context = [], [], {}
    for word in words:
        decomposition = unicodedata.normalize('NFD',word)
        missing = [c for c in decomposition if c not in vocab]
        roundtrip = not missing and decode([vocab[c] for c in decomposition],tokens)[0] == word
        alt = variants.get(word)
        pool = available[word] + (available[alt] if alt else [])
        clips = selected_records(pool)
        clips += [row for row in local if ''.join(row['text'].split()) in {word,alt} and row['seconds'] <= 4.5]
        for row in clips:
            selected.append({**row,'curriculum_word':word})
        contexts = context_available[word] + (context_available[alt] if alt else [])
        context_selection = []
        if not clips:
            rng = random.Random(20261030)
            for age in (3,4,5,6):
                age_rows = sorted([r for r in contexts if r['age']==age],key=lambda r:r['record_id'])
                if age_rows:
                    context_selection.append(rng.choice(age_rows))
            for row in context_selection:
                existing = selected_context.setdefault(row['record_id'],{**row,'curriculum_words':[]})
                if word not in existing['curriculum_words']:
                    existing['curriculum_words'].append(word)
        word_results.append({'word':word,'stages':[s['id'] for s in stages if word in s['targets']],
                             'token_roundtrip_ok':roundtrip,'missing_tokens':missing,
                             'original_form':form_stats[word], 'alternative_form':alt,
                             'alternative_counts':form_stats[alt] if alt else None,
                             'available_dev_full_utterances':len(pool),'selected_audio':len(clips),
                             'available_dev_context_utterances':len(contexts),'selected_context_clips':len(context_selection),
                             'selected_audio_by_group':dict(Counter(r['group'] for r in clips))})
    phonemes = {word:games.g2p.to_ipa(word) for word in forms}
    scores = {spoken:{target:games.matcher.score_against(phonemes[spoken],phonemes[target])[1] for target in words} for spoken in forms}
    stage_risks = []
    for stage in stages:
        for spoken in words:
            if spoken in stage['targets']:
                continue
            highest = max(scores[spoken][target] for target in stage['targets'])
            if highest >= .85:
                family = next((family for family, values in FAMILIES.items() if spoken in [normalize(w) for w in values.split()]),spoken)
                winners = [word for word in stage['targets'] if abs(scores[spoken][word]-highest)<1e-9]
                stage_risks.append({'stage':stage['id'],'spoken':spoken,'best_targets':winners,'score':highest,
                                    'same_action_unlisted':family == stage['family'],'ties':len(winners)>1})
    ideal_checks = []
    for stage in stages:
        for spoken in stage['targets']:
            highest = max(scores[spoken][target] for target in stage['targets'])
            winners = [word for word in stage['targets'] if abs(scores[spoken][word]-highest)<1e-9]
            ideal_checks.append({'stage':stage['id'],'spoken':spoken,'leaders':winners,'score':highest,'ambiguous':len(winners)>1})
    summary = {'created_at':datetime.now().astimezone().isoformat(), 'activities':{a:{'steps':9,'entries':sum(len(s['targets']) for s in stages if s['activity']==a),'unique_words':len({w for s in stages if s['activity']==a for w in s['targets']})} for a in ('music','dinosaur','plant')},
               'stages':27,'entries':199,'unique_words':len(words),'all_token_roundtrips':all(r['token_roundtrip_ok'] for r in word_results),
               'manifest_scan':scan,'no_training_boundary_original':[r['word'] for r in word_results if not r['original_form']['train']['boundary_utterances']],
               'no_training_boundary_with_spelling_variant':[r['word'] for r in word_results if not r['original_form']['train']['boundary_utterances'] and not (r['alternative_counts'] and r['alternative_counts']['train']['boundary_utterances'])],
               'no_training_standalone_original':[r['word'] for r in word_results if not r['original_form']['train']['standalone_utterances']],
               'no_audio_to_evaluate':[r['word'] for r in word_results if not r['selected_audio']],
               'no_isolated_or_context_audio':[r['word'] for r in word_results if not r['selected_audio'] and not r['selected_context_clips']],
               'context_audio_clips_selected':len(selected_context),
               'audio_clips_selected':len(selected),'audio_by_group':dict(Counter(r['group'] for r in selected)),
               'ideal_stage_ties':[r for r in ideal_checks if r['ambiguous']], 'ideal_out_of_stage_collisions':len(stage_risks),
               'method':'Frozen train/dev manifests only; C-drive only; spaces and terminal punctuation normalized for counting. Boundary counts are utterances containing the complete requested form, not substring counts. All words are representable, which does not guarantee recognition. Alternative 거예요 counted separately. No final test used. Source list preserved including music-7 omission.'}
    for name,data in [('catalog.json',stages),('summary.json',summary),('word-coverage.json',word_results),('form-counts.json',form_stats),
                      ('word-pair-scores.json',scores),('stage-phonetic-risks.json',stage_risks),('ideal-stage-checks.json',ideal_checks),('selected-audio.json',selected),
                      ('selected-context-audio.json',list(selected_context.values()))]:
        write(output/name,data)
    print(json.dumps({'phase':'complete','output':str(output),**{k:v for k,v in summary.items() if k not in ('method','manifest_scan')}},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
