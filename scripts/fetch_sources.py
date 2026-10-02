"""Retrieve pinned upstream inputs without redistributing them in this repository."""
from pathlib import Path
import json,hashlib,urllib.request
P=Path(__file__).resolve().parents[1]
m=json.loads((P/'SOURCE_DATA.json').read_text())
base=P/'evidence/eleven_review_remediation_2026-09-21_001619'
for rel,digest in m['files'].items():
 tail=rel.split('sources/gollum/',1)[1]
 dest=base/rel
 if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==digest:continue
 url=f"https://raw.githubusercontent.com/schwallergroup/gollum/{m['commit']}/{tail}"
 data=urllib.request.urlopen(url,timeout=90).read()
 if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Source checksum mismatch: '+rel)
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
print('Pinned chemistry inputs present and checksums verified.')

# METR website inputs are retrieved, not redistributed.
for item in json.loads((P/'METR_INPUTS.json').read_text()):
 dest=P/item['path']
 if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==item['sha256']:continue
 data=urllib.request.urlopen(item['url'],timeout=90).read()
 if hashlib.sha256(data).hexdigest()!=item['sha256']:raise RuntimeError('METR source changed; expected pinned bytes: '+item['path'])
 dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
print('METR input checksums verified.')
