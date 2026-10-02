"""Refresh mechanically checkable artifact facts without overwriting literature judgments."""
from pathlib import Path
import hashlib,json,re
import fitz
P=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
doc=fitz.open(P/'manuscript/paper.pdf')
main=(P/'manuscript/body.tex').read_text();entry=(P/'manuscript/paper.tex').read_text();app=(P/'manuscript/appendix.tex').read_text()
keys=set()
for arg in re.findall(r'\\cite[a-z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}',entry+main+app):keys.update(x.strip() for x in arg.split(','))
refs=next(i+1 for i,p in enumerate(doc) if re.search(r'(?m)^References\s*$',p.get_text()))
mainlast=next(i+1 for i,p in enumerate(doc) if 'AI use disclosure' in p.get_text())
appstart=next(i+1 for i,p in enumerate(doc) if 'Public-data prediction tests and reproduction' in p.get_text())
info=dict(pdf_sha256=sha(P/'manuscript/paper.pdf'),pdf_pages=len(doc),main_last_page=mainlast,references_first_page=refs,appendix_first_page=appstart,main_sections=len(re.findall(r'\\section\{',main)),main_figures=len(re.findall(r'\\begin\{figure\}',main)),main_tables=len(re.findall(r'\\begin\{table\}',main)),main_numbered_equations=len(re.findall(r'\\begin\{equation\}',main)),appendix_figures=len(re.findall(r'\\begin\{figure\}',app)),appendix_tables=len(re.findall(r'\\begin\{table\}',app)),appendix_sections=len(re.findall(r'\\section\{',app)),appendix_numbered_equations=len(re.findall(r'\\begin\{equation\}',app)),distinct_cited_keys=len(keys),cited_keys=sorted(keys),visual_verification='Hash-bound dated review required; build alone does not certify appearance.')
(P/'evidence/current_artifact_inventory.json').write_text(json.dumps(info,indent=2)+'\n')
reg=P/'Comparators/COMPARATOR_REGISTRY.md';s=reg.read_text() if reg.exists() else '<!-- ACTIVE_ARTIFACT_START --><!-- ACTIVE_ARTIFACT_END -->';a='<!-- ACTIVE_ARTIFACT_START -->';b='<!-- ACTIVE_ARTIFACT_END -->'
block=f'''{a}
### Current artifact inventory

Canonical PDF: `manuscript/paper.pdf`; SHA-256 `{info['pdf_sha256']}`; {len(doc)} PDF pages. Main text and disclosure end on page {mainlast}; references begin on page {refs}; appendix starts page {appstart}.

MAIN_PAPER: {info['main_sections']} sections; {info['main_figures']} figure (one panel in the current draft); {info['main_tables']} tables; {info['main_numbered_equations']} numbered equations; no labeled theorem/proposition/lemma. Supplement: {info['appendix_sections']} sections, {info['appendix_numbered_equations']} numbered equations, {info['appendix_figures']} figure and {info['appendix_tables']} table. {len(keys)} distinct references are cited across main and supplement.

Source-derived counts and PDF section boundaries: `evidence/current_artifact_inventory.json`. A build updates facts; visual and substantive review require matching dated reviews and are not inferred from build success.
{b}'''
if a not in s or b not in s:raise RuntimeError('Canonical registry inventory markers missing')
if reg.exists():reg.write_text(s.split(a,1)[0]+block+s.split(b,1)[1])
print(json.dumps(info,indent=2))
