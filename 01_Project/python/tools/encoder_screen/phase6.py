"""Final bounded BERT diagnostic and legacy deployment of trained finalists."""
import time
import traceback
from common import WORK,load,dump
from deploy import convert,player
from legacy import convert as legacy_convert,player as legacy_player

if __name__=='__main__':
    while not (WORK/'phase5-finished.json').exists():time.sleep(10)
    results=[]
    folded=WORK/'exports/w2vbert-shape-folded/folding.json'
    if folded.exists() and all(r['ctc_equal'] for r in load(folded)['ort_vs_pytorch']):
        key='w2vbert-shape-folded'
        for version,importer,runner in [('unity6',convert,player),('unity2022',legacy_convert,legacy_player)]:
            try:
                print('BERT_FOLDED',version,flush=True)
                imported=importer(key)
                if not imported.get('success'):raise RuntimeError(str(imported))
                result={'key':key,'version':version,'import':imported,'player':runner(key)}
            except Exception:result={'key':key,'version':version,'error':traceback.format_exc()}
            results.append(result);dump(WORK/'final-compatibility-results.json',results)
    for _,base in load(WORK/'trained-deployment-selection.json')['selected']:
        key=base+'-pilot917'
        try:
            print('TRAINED_LEGACY',key,flush=True)
            imported=legacy_convert(key)
            if not imported.get('success'):raise RuntimeError(str(imported))
            result={'key':key,'version':'unity2022','import':imported,'player':legacy_player(key)}
        except Exception:result={'key':key,'version':'unity2022','error':traceback.format_exc()}
        results.append(result);dump(WORK/'final-compatibility-results.json',results)
    dump(WORK/'phase6-finished.json',{'finished':True})
