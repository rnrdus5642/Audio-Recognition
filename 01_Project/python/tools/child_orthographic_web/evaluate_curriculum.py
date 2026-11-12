"""Replay user-supplied stage targets using the deployed model, without edits."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import re

import numpy as np

from audit_curriculum import match_forms, noun_particle_hits
from evaluate_performance import Evaluator, ROOT, RATE, safe, write, waveform, encoded, normalize
from word_game import StartWordGame


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = safe(args.source)
    output = source / 'audio-evaluation'
    ev = Evaluator(output)
    baseline = ROOT / '04_Models_and_Experiments/orthographic-jamo-20261027/web-test/performance-20261030-025230'
    caches = [baseline/'predictions.jsonl',baseline/'decision-diagnosis-20261030-032424/predictions.jsonl']
    for cache in caches:
        if cache.exists():
            for line in cache.read_text(encoding='utf-8').splitlines():
                item = json.loads(line)
                assert item['model_id'] == ev.status['model_id']
                ev.cache[item['audio_hash']] = item['prediction']
    stages = load(source/'catalog.json')
    selected = load(source/'selected-audio.json')
    contexts = load(source/'selected-context-audio.json')
    full, trials = [], []
    print(json.dumps({'phase':'start','isolated_audio':len(selected),'context_audio':len(contexts)},ensure_ascii=False),flush=True)
    for number,row in enumerate(selected,1):
        wave = waveform(row)
        prediction = ev.infer(wave)
        full.append({**row,'prediction':prediction,'exact_reference':normalize(prediction['text'])==normalize(row['text'])})
        sessions = {stage['id']:ev.games.start(StartWordGame(targets=stage['raw_targets']))['session_id'] for stage in stages}
        results = {}
        aggregate = {stage['id']:{'state':'listening','streak':0,'seconds':None,'text':None} for stage in stages}
        signal = np.zeros(RATE*10,dtype=np.float32)
        signal[6400:6400+len(wave)] = wave
        for step in range(1,21):
            end = step*8000
            chunk = signal[max(0,end-40000):end]
            chunk = np.pad(chunk,(40000-len(chunk),0))
            ev.pending = ev.infer(chunk)
            canonical = encoded(chunk)
            for stage in stages:
                sid = stage['id']
                if sid in results:
                    continue
                answer = ev.games.frame(sessions[sid],step,step*.5,chunk,canonical,RATE)
                group = aggregate[sid]
                if group['state']=='listening':
                    # All candidates in this stage are allowed. Ties and
                    # transitions between them need not reset an action streak.
                    group['streak'] = group['streak']+1 if answer['similarity']>=.85 else 0
                    if group['streak']>=2:
                        group.update(state='accepted',seconds=step*.5,text=answer['prediction']['text'])
                if answer['state']!='listening':
                    results[sid] = answer
        for stage in stages:
            sid = stage['id']
            if sid not in results:
                results[sid] = ev.games.finish(sessions[sid],'timeout')
            if aggregate[sid]['state']=='listening':
                aggregate[sid]['state']='retry'
            results[sid]['trace'] = ev.games.get(sessions[sid])['trace']
            trials.append({'record_id':row['record_id'],'reference':row['text'],'curriculum_word':row['curriculum_word'],
                           'group':row['group'],'age':row['age'],'stage':sid,
                           'is_positive':row['curriculum_word'] in stage['targets'],
                           'current':results[sid],'stage_streak_counterfactual':aggregate[sid]})
        if number%10==0:
            print(json.dumps({'phase':'isolated','done':number,'total':len(selected),'new_gpu_calls':ev.calls}),flush=True)
    write(output/'isolated-asr.json',full)
    with gzip.open(output/'stage-trials.json.gz','wt',encoding='utf-8') as stream:
        json.dump(trials,stream,ensure_ascii=False)
    forms = load(source/'form-counts.json')
    trie = {}
    for word in forms:
        node = trie
        for char in word:
            node = node.setdefault(char,{})
        node[''] = word
    context_results = []
    for index,row in enumerate(contexts,1):
        prediction = ev.infer(waveform(row))
        _,_,boundary = match_forms(prediction['text'],trie)
        heard = boundary | noun_particle_hits(prediction['text'])
        target_results = {word:word in heard or word.replace('거에요','거예요') in heard for word in row['curriculum_words']}
        context_results.append({**row,'prediction':prediction,'target_preserved':target_results})
        if index%25==0:
            print(json.dumps({'phase':'context','done':index,'total':len(contexts)}),flush=True)
    write(output/'context-asr.json',context_results)
    stage_results = []
    for stage in stages:
        pos = [r for r in trials if r['stage']==stage['id'] and r['is_positive']]
        neg = [r for r in trials if r['stage']==stage['id'] and not r['is_positive']]
        stage_results.append({**stage,'positive_clips':len(pos),'tested_positive_words':sorted({r['curriculum_word'] for r in pos}),
                              'positive_by_group':dict(Counter(r['group'] for r in pos)),
                              'positive_current_accepted':sum(r['current']['state']=='accepted' for r in pos),
                              'positive_stage_streak_accepted':sum(r['stage_streak_counterfactual']['state']=='accepted' for r in pos),
                              'negative_clips':len(neg), 'negative_current_accepted':sum(r['current']['state']=='accepted' for r in neg),
                              'negative_stage_streak_accepted':sum(r['stage_streak_counterfactual']['state']=='accepted' for r in neg)})
    words = sorted({word for stage in stages for word in stage['targets']})
    word_results = []
    for word in words:
        clips = [r for r in full if r['curriculum_word']==word]
        c = [r for r in context_results if word in r['curriculum_words']]
        by_stage = {}
        for stage in stages:
            if word in stage['targets']:
                records=[r for r in trials if r['stage']==stage['id'] and r['curriculum_word']==word]
                by_stage[stage['id']] = {'clips':len(records),'current_accepted':sum(r['current']['state']=='accepted' for r in records),
                                        'stage_streak_accepted':sum(r['stage_streak_counterfactual']['state']=='accepted' for r in records)}
        word_results.append({'word':word,'isolated_clips':len(clips),'isolated_exact_text':sum(r['exact_reference'] for r in clips),
                             'isolated_predictions':dict(Counter(r['prediction']['text'] for r in clips)),
                             'by_stage':by_stage,'context_clips':len(c),'context_preserved':sum(r['target_preserved'][word] for r in c)})
    summary = {'model':ev.status,'unique_isolated_clips':len(full),'isolated_words':sum(r['isolated_clips']>0 for r in word_results),
               'unique_context_clips':len(context_results),'context_words':sum(r['context_clips']>0 for r in word_results),
               'stage_trials':len(trials),'new_gpu_calls':ev.calls,'cache_hits':ev.hit_count,
               'positive_trials':sum(r['is_positive'] for r in trials),
               'positive_current_accepted':sum(r['is_positive'] and r['current']['state']=='accepted' for r in trials),
               'positive_stage_streak_accepted':sum(r['is_positive'] and r['stage_streak_counterfactual']['state']=='accepted' for r in trials),
               'negative_current_accepted':sum(not r['is_positive'] and r['current']['state']=='accepted' for r in trials),
               'negative_stage_streak_accepted':sum(not r['is_positive'] and r['stage_streak_counterfactual']['state']=='accepted' for r in trials),
               'limits':'Stage trials repeat the same audio across stages, not independent tests. Positives are selected dev single-word utterances plus preserved local recordings of unknown age and incomplete label review. Context tests measure preservation in full sentences, not isolated-word game performance. Negative trials are intentional out-of-list word probes, not natural-use false acceptance rate. Counterfactual stage streak never deployed.'}
    write(output/'summary.json',summary)
    write(output/'stage-results.json',stage_results)
    write(output/'word-results.json',word_results)
    print(json.dumps({'phase':'complete',**{k:v for k,v in summary.items() if k not in ('model','limits')}},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
