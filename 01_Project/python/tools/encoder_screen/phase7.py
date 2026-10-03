"""Check the held-out owned words on the actual selected Unity GPU graphs."""
from pathlib import Path
import sys
import time
from common import WORK,load,dump
from deploy import command

if __name__=='__main__':
    while not (WORK/'phase6-finished.json').exists():time.sleep(10)
    results=[]
    for _,key in load(WORK/'trained-deployment-selection.json')['selected']:
        print('DEPLOYED_WORDS',key,flush=True)
        rc=command([sys.executable,'-X','utf8','-u',Path(__file__).with_name('deployed_words.py'),key],
                    WORK/'logs'/f'deployed-words-{key}.log',timeout=1800)
        results.append({'key':key,'exit_code':rc});dump(WORK/'deployed-word-results.json',results)
        print('DEPLOYED_WORDS_RESULT',key,rc,flush=True)
    dump(WORK/'phase7-finished.json',{'finished':True})
