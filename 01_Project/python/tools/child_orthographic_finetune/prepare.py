"""Verify every selected WAV and publish immutable, hash-bound manifests."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import json
import random
import re
import time
import wave

from common import ROOT, PREP, WORK, MANIFESTS, POLICY, read, dump, sha, normalize, encode, frames, ctc_width, safe_audio


def verify(row):
    path = safe_audio(row['audio_path'])
    before = path.stat()
    payload = path.read_bytes()
    expected = row.get('audio_sha256') or row['retained_sha256_at_cleanup']
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected or len(payload) != row['audio_bytes']:
        raise ValueError(f'WAV differs from verified source: {path}')
    with wave.open(io.BytesIO(payload),'rb') as wav:
        if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getcomptype()) != (16000,1,2,'NONE'):
            raise ValueError(f'Unexpected WAV format: {path}')
        count = wav.getnframes()
        pcm = wav.readframes(count)
        if len(pcm) != count*2:
            raise ValueError(f'Truncated PCM: {path}')
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError(f'WAV changed while validating: {path}')
    text = normalize(row['label_original'])
    assert text == row['label_text'] and row['label_status']=='usable_orthography'
    ids = encode(text)
    result = {k:row[k] for k in ('dataset_id','record_id','age','speaker_key','split','audio_path')}
    result.update(text=text, label_ids=ids, samples=count, seconds=count/16000,
                  audio_sha256=actual, audio_bytes=len(payload), mtime_ns=after.st_mtime_ns)
    reason = None
    if frames(count)<ctc_width(ids):
        reason = 'ctc_infeasible_exact_header'
    elif row['split']=='train' and count>320000:
        reason = 'train_over_20_seconds'
    if 'audio_details' in row:
        assert count==row['audio_details']['frames']
    return result,reason,hashlib.sha256(pcm).hexdigest()


def monitor_selection(dev):
    selected=[]
    rng=random.Random(917)
    for age in (3,4,5,6):
        speakers=defaultdict(list)
        for row in dev:
            if row['age']==age:
                speakers[row['speaker_key']].append(row)
        for seq in speakers.values():
            rng.shuffle(seq)
        keys=sorted(speakers)
        rng.shuffle(keys)
        seconds=0
        while seconds<1800:
            moved=False
            for key in keys:
                if speakers[key]:
                    item=speakers[key].pop()
                    selected.append(item)
                    seconds+=item['seconds']
                    moved=True
                    if seconds>=1800:
                        break
            assert moved
    return selected


def main():
    if (MANIFESTS/'report.json').exists():
        report=read(MANIFESTS/'report.json')
        assert report['status']=='passed'
        assert all(sha(MANIFESTS/name)==digest for name,digest in report['manifest_hashes'].items())
        print('Validated manifests already exist; preserved unchanged.',flush=True)
        return
    MANIFESTS.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    assert not any((MANIFESTS/f'{p}.jsonl.gz').exists() for p in ('train','dev','test','monitor'))
    plan=read(PREP/'dataset-plan.json')
    exclusion=read(PREP/plan['exclusion_policy'])
    assert sha(PREP/exclusion['exclusion_manifest'])==exclusion['exclusion_manifest_sha256']
    with gzip.open(PREP/exclusion['exclusion_manifest'],'rt',encoding='utf-8') as stream:
        denied={(r['dataset_id'],r['record_id']) for r in map(json.loads,stream)}
    pattern=re.compile(exclusion['tag_pattern'])
    speakers=read(PREP/'speaker-splits.json')['speakers']
    audit=read(ROOT/'99_Organization/annotation-tag-audit-20261027/report.json')
    source_hashes={item['path']:item['sha256'] for item in audit['inputs']}
    assert all(sha(ROOT/p)==h for p,h in source_hashes.items())
    frozen={name:sha(PREP/name) for name in ('vocab.json','label-policy.json','speaker-splits.json','exclusion-policy.json')}
    counts=Counter(); seconds=Counter(); excluded=Counter(); ages=defaultdict(Counter)
    seen=set(); split_speakers=defaultdict(set); wave_splits={}; pcm_splits={}; duplicate_pcm=Counter()
    dev=[]; longest={}; seen_count=0; start=time.monotonic()
    def incoming(partition):
        for directory in plan['active_catalogs'].values():
            with gzip.open(PREP/directory/f'{partition}.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    row=json.loads(line)
                    assert row['age'] in (3,4,5,6)
                    assert row['split']==partition==speakers[row['speaker_key']]['split']
                    assert row['age']==speakers[row['speaker_key']]['age']
                    assert row['label_policy_version']==POLICY['version']
                    assert not pattern.search(row['label_original'])
                    assert (row['dataset_id'],row['record_id']) not in denied
                    assert not row['review_reasons']
                    yield row
    with ThreadPoolExecutor(max_workers=8) as executor, gzip.open(MANIFESTS/'resource-excluded.jsonl.gz','wt',encoding='utf-8') as rejected:
        for partition in ('train','dev','test'):
            source=iter(incoming(partition))
            with gzip.open(MANIFESTS/f'{partition}.jsonl.gz','xt',encoding='utf-8') as dest:
                while True:
                    batch=[]
                    for _ in range(512):
                        try:
                            batch.append(next(source))
                        except StopIteration:
                            break
                    if not batch:
                        break
                    for row,reason,pcm_hash in executor.map(verify,batch):
                        seen_count+=1
                        key=(row['dataset_id'],row['record_id'])
                        assert key not in seen
                        seen.add(key)
                        if reason:
                            rejected.write(json.dumps({**row,'exclusion_reason':reason},ensure_ascii=False)+'\n')
                            excluded[reason]+=1
                            continue
                        # Same-speaker partitions are frozen. Byte-identical audio across partitions is a hard stop.
                        for digest, registry in ((row['audio_sha256'],wave_splits),(pcm_hash,pcm_splits)):
                            previous=registry.setdefault(digest,partition)
                            assert previous==partition, f'Cross-partition duplicate audio: {row["audio_path"]}'
                        duplicate_pcm[(partition,pcm_hash)]+=1
                        split_speakers[partition].add(row['speaker_key'])
                        counts[partition]+=1
                        seconds[partition]+=row['seconds']
                        ages[partition][str(row['age'])]+=1
                        if partition not in longest or row['samples']>longest[partition]['samples']:
                            longest[partition]=row
                        dest.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
                        if partition=='dev':
                            dev.append(row)
                    if seen_count%10240==0:
                        progress={'status':'verifying_wavs','verified':seen_count,'counts':dict(counts),'elapsed_seconds':round(time.monotonic()-start,1)}
                        dump(WORK/'preparation-progress.json',progress)
                        print(json.dumps(progress),flush=True)
    assert seen_count==879549
    assert all(not(split_speakers[a]&split_speakers[b]) for a in split_speakers for b in split_speakers if a<b)
    monitor=monitor_selection(dev)
    with gzip.open(MANIFESTS/'monitor.jsonl.gz','xt',encoding='utf-8') as dest:
        for row in monitor:
            dest.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
    assert all(sha(PREP/name)==digest for name,digest in frozen.items())
    assert all(sha(ROOT/p)==h for p,h in source_hashes.items())
    report={'status':'passed','date_kst':'2026-10-27','all_wavs_sha256_and_exact_header_verified':True,
            'source_catalog_hashes':source_hashes,'frozen_input_hashes':frozen,
            'manifest_hashes':{p.name:sha(p) for p in MANIFESTS.glob('*.jsonl.gz')},
            'candidate_records_checked':seen_count,'counts':dict(counts),'hours':{k:v/3600 for k,v in seconds.items()},
            'age_counts':dict(ages),'resource_exclusions':dict(excluded),'longest':longest,
            'duplicate_pcm_within_partition_extra_records':sum(n-1 for n in duplicate_pcm.values() if n>1),
            'cross_partition_speaker_overlap':0,'cross_partition_audio_overlap':0,
            'annotation_excluded_records':len(denied),'monitor_records':len(monitor),
            'monitor_hours_by_age':{str(age):sum(r['seconds'] for r in monitor if r['age']==age)/3600 for age in (3,4,5,6)},
            'elapsed_seconds':time.monotonic()-start}
    dump(MANIFESTS/'report.json',report)
    dump(WORK/'preparation-progress.json',{'status':'complete','manifest_report':str(MANIFESTS/'report.json')})
    print(json.dumps({k:report[k] for k in ('status','counts','hours','resource_exclusions','elapsed_seconds')}),flush=True)


if __name__=='__main__':
    main()
