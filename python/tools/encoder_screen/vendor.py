"""Fetch only the reviewed, hash-pinned ESPnet module sources in vendor-lock."""
from pathlib import Path
import urllib.request
from common import WORK, load, dump, digest


def ensure():
    lock=load(Path(__file__).with_name('vendor-lock.json'))
    directory=WORK/'vendor'; directory.mkdir(parents=True,exist_ok=True)
    for row in lock['files']:
        target=directory/row['file']
        if not target.exists():
            url=f"https://raw.githubusercontent.com/{lock['repo']}/{lock['revision']}/{row['source_path']}"
            target.write_bytes(urllib.request.urlopen(url,timeout=90).read())
        assert digest(target)==row['sha256'],target
    dump(directory/'espnet-main-revision.json',{k:lock[k] for k in ['repo','revision']})


if __name__=='__main__':ensure()
