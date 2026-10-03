"""Early functional-only check while WavLM trains; timings are excluded."""
import time
import traceback
from common import WORK,load,dump
from deploy import convert,player
from legacy import convert as legacy_convert,player as legacy_player

if __name__=='__main__':
    # The preceding legacy timing runs must have ended before adding GPU work.
    while not (WORK/'finetunes/wavlm/seed917/config.json').exists():time.sleep(5)
    results=[];key='w2vbert-shape-folded'
    for version,importer,runner in [('unity6',convert,player),('unity2022',legacy_convert,legacy_player)]:
        try:
            imported=importer(key)
            if not imported.get('success'):raise RuntimeError(str(imported))
            result={'key':key,'version':version,'import':imported,
                    'player':runner(key,suffix='-functional-concurrent')}
        except Exception:result={'key':key,'version':version,'error':traceback.format_exc()}
        results.append(result);dump(WORK/'bert-folded-smoke.json',results)
        actual=result.get('player',{}).get('result') or {}
        print('BERT_FOLDED_SMOKE',version,actual.get('success'),actual.get('mismatch_frames'),actual.get('error'),flush=True)
