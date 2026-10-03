"""Collect aggregate evidence without copying audio or source transcript lists."""
from collections import Counter,defaultdict
from pathlib import Path
import argparse
import numpy as np
from common import ROOT,WORK,MODELS,dump,load,digest


def maybe(path):
    return load(path) if Path(path).exists() else None


def selected_probe(key):
    """Select from raw/scalar and both LRs using only validation scores."""
    choices=[]
    for folder,kind in [('probes','raw'),('probes-normalized','single_scalar')]:
        path=WORK/folder/key/'result.json'
        if path.exists():
            r=load(path)
            choices.append((r['selected']['valid']['jer'],kind,path.parent,r))
    if not choices:return None,None
    _,kind,directory,result=min(choices,key=lambda x:x[0])
    return {'kind':kind,'all_four_conditions_complete':len(choices)==2,
            'rule':'minimum valid JER over raw/scalar x two learning rates',
            'selected':result['selected'],'test':result['test'],
            'head_sha256':digest(directory/'head.pt')},load(directory/'test-predictions.json')


def domain_summary(domain):
    """Keep group/word totals, without source paths, speaker IDs or clip IDs."""
    output={k:v for k,v in domain.items() if k!='records'}
    groups=defaultdict(list)
    for r in domain['records']:groups[(r['condition'],r['label'])].append(r)
    output['by_word']=[{'condition':condition,'word':word,'recordings':len(rows),
                       'exact':sum(r['errors']==0 for r in rows),
                       'ae_merged_exact':sum(r['ae_merged_errors']==0 for r in rows),
                       'jer':sum(r['errors'] for r in rows)/sum(r['reference_jamo'] for r in rows),
                       'hypotheses':dict(Counter(r['hyp_jamo'] for r in rows))}
                      for (condition,word),rows in sorted(groups.items())]
    output['combined']=[{'condition':condition,'recordings':len(rows),
                         'exact':sum(r['errors']==0 for r in rows),
                         'ae_merged_exact':sum(r['ae_merged_errors']==0 for r in rows),
                         'jer':sum(r['errors'] for r in rows)/sum(r['reference_jamo'] for r in rows)}
                        for condition in ['clean','pad_1s','noise_20db','noise_10db']
                        if (rows := [r for r in domain['records'] if r['condition']==condition])]
    return output


def bootstrap(predictions, baseline, manifest, rounds=5000):
    by_id={r['id']:r for r in manifest['rows'] if r['split']=='test'}
    speakers=sorted({r['speaker'] for r in by_id.values()})
    idx={s:i for i,s in enumerate(speakers)}
    a=np.zeros((len(speakers),3),dtype=np.float64)
    b={r['id']:r for r in baseline}
    assert set(r['id'] for r in predictions)==set(b)==set(by_id)
    for p in predictions:
        j=idx[by_id[p['id']]['speaker']]
        a[j]+=[p['errors'],b[p['id']]['errors'],p['reference_jamo']]
    rng=np.random.default_rng(9142026)
    samples=a[rng.integers(0,len(a),(rounds,len(a)))].sum(axis=1)
    difference=(samples[:,0]-samples[:,1])/samples[:,2]
    return {'candidate_minus_baseline_jer':float((a[:,0].sum()-a[:,1].sum())/a[:,2].sum()),
            'percentile_95':np.percentile(difference,[2.5,97.5]).tolist(),
            'speaker_clusters':len(speakers),'rounds':rounds,'seed':9142026,
            'note':'Exploratory paired speaker bootstrap, no multiple-model selection correction; not population-wide certainty'}


def collect():
    manifest=load(WORK/'manifest.json')
    counts=Counter(i for r in manifest['rows'] if r['split']=='train' for i in r['label_ids'])
    data={'date':'2026-09-14','environment':maybe(WORK/'environment.json'),
          'contract':load(WORK/'contract.json'),'manifest_sha256':digest(WORK/'manifest.json'),
          'train_jamo_counts':{s:counts[i] for s,i in manifest['vocab'].items() if i},
          'candidates':{},'notes':['No production model/runtime/package was replaced by this harness',
                                 'Frozen-probe and short fine-tuning scores are not convergence rankings',
                                 'Derived transcript labels do not validate pronunciation-error diagnoses']}
    base=maybe(WORK/'probes/xlsr300/test-predictions.json')
    normalized_base=maybe(WORK/'probes-normalized/xlsr300/test-predictions.json')
    _,selected_base=selected_probe('xlsr300')
    finetuned_base=maybe(WORK/'finetunes/xlsr300/seed917/test-predictions.json')
    test_rows={r['id']:r for r in manifest['rows'] if r['split']=='test'}
    def strata(predictions):
        result={}
        for field in ['age','style','noise','device']:
            groups=defaultdict(list)
            for p in predictions:groups[test_rows[p['id']][field]].append(p)
            result[field]={s:{'recordings':len(v),'jer':sum(x['errors'] for x in v)/sum(x['reference_jamo'] for x in v)}
                           for s,v in groups.items()}
        return result
    for key,spec in MODELS.items():
        model=WORK/'models'/key
        r={'spec':spec,'download':maybe(model/'verified.json'),
           'features':maybe(WORK/'features'/key/'metadata.json'),
           'frozen_probe':maybe(WORK/'probes'/key/'result.json')}
        # The download record is small and contains only public checkpoint provenance.
        export=maybe(WORK/'exports'/key/'export.json')
        if export:
            r['export']={k:v for k,v in export.items() if k not in ['weights','source']}
            r['export']['external_weight_bytes']=sum((WORK/'exports'/key/v['file']).stat().st_size for v in export['weights'])
            r['export']['weight_tensor_count']=len(export['weights'])
        r['unity6_import']=maybe(WORK/'exports'/key/'model.sentis.import.json')
        r['unity6_player']=maybe(WORK/'exports'/key/'unity-player.json')
        r['unity2022_import']=maybe(WORK/'exports'/key/'model-sentis1.sentis.import.json')
        r['unity2022_player']=maybe(WORK/'exports'/key/'unity2022-player.json')
        r['finetunes']={}
        r['selected_probe'],selected_predictions=selected_probe(key)
        if selected_predictions:
            r['selected_probe']['strata']=strata(selected_predictions)
            if selected_base:
                r['selected_probe']['vs_xlsr300']=bootstrap(selected_predictions,selected_base,manifest)
        r['normalized_probe']=maybe(WORK/'probes-normalized'/key/'result.json')
        if r['normalized_probe']:
            pred=load(WORK/'probes-normalized'/key/'test-predictions.json')
            r['normalized_strata']=strata(pred)
            if normalized_base:r['normalized_vs_xlsr300']=bootstrap(pred,normalized_base,manifest)
        if r['frozen_probe']:
            r['frozen_strata']=strata(load(WORK/'probes'/key/'test-predictions.json'))
        for f in (WORK/'finetunes'/key).glob('seed*/result.json'):
            value=load(f)
            predictions=load(f.parent/'test-predictions.json')
            value['test_strata']=strata(predictions)
            if finetuned_base:
                value['vs_xlsr300_seed917']=bootstrap(predictions,finetuned_base,manifest)
            if selected_predictions:
                value['vs_selected_frozen_head']=bootstrap(predictions,selected_predictions,manifest)
            for kind in ['initial','final']:
                domain=maybe(f.parent/f'domain-{kind}.json')
                if domain:value[f'domain_{kind}']=domain_summary(domain)
            r['finetunes'][f.parent.name]=value
        if r['frozen_probe'] and base:
            pred=load(WORK/'probes'/key/'test-predictions.json')
            r['frozen_vs_xlsr300']=bootstrap(pred,base,manifest)
        data['candidates'][key]=r
    data['trained_deployment']=maybe(WORK/'trained-deployment-results.json')
    data['trained_pairwise']={}
    for candidate,baseline in [('hubertko1b','hubertko'),('w2vbert','hubertko'),('wavlm','hubertko')]:
        a=maybe(WORK/'finetunes'/candidate/'seed917/test-predictions.json')
        b=maybe(WORK/'finetunes'/baseline/'seed917/test-predictions.json')
        if a and b:data['trained_pairwise'][candidate+'_minus_'+baseline]=bootstrap(a,b,manifest)
    data['second_seed_selection']=maybe(WORK/'second-seed-selection.json')
    data['label_audit']=maybe(WORK/'label-audit.json')
    data['normalization_decision']=maybe(WORK/'normalization-decision.json')
    data['final_compatibility']=maybe(WORK/'final-compatibility-results.json')
    data['completed_phases']={str(i):(WORK/f'phase{i}-finished.json').exists() for i in range(2,8)}
    dump(WORK/'aggregate.json',data)
    print('COLLECTED',[(k,round(r['frozen_probe']['test']['jer']*100,2) if r['frozen_probe'] else None,
                       r['unity6_player'].get('success') if r['unity6_player'] else None)
                       for k,r in data['candidates'].items()])
    return data


if __name__=='__main__':collect()
