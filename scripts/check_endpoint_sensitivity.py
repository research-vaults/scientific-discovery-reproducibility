"""Independently reconstruct all new aggregate losses using the standard library."""
from pathlib import Path
from collections import defaultdict
import csv,json,math
P=Path(__file__).resolve().parents[1]
R=P/'evidence/consequential_gap_repairs_2026-09-26_012131/results'
rows=list(csv.DictReader((R/'predictions.csv').open()));acc=defaultdict(list)
for x in rows:
 y=int(x['observed']);p=max(1e-6,min(1-1e-6,float(x['predicted'])))
 losses=((p-y)**2,-y*math.log(p)-(1-y)*math.log1p(-p))
 for st in ['all','not_yet_hit']:
  if st=='not_yet_hit' and x['hit_already']=='True':continue
  acc[tuple(x[k] for k in ['quantile','pilot','budget','predictor','family','table'])+(st,)].append(losses)
fam=defaultdict(list)
for k,v in acc.items():fam[k[:5]+(k[-1],)].append([sum(z[i] for z in v)/len(v) for i in range(2)])
agg=defaultdict(list)
for k,v in fam.items():agg[k[:4]+(k[-1],)].append([sum(z[i] for z in v)/len(v) for i in range(2)])
mx=0;count=0
for x in csv.DictReader((R/'summary.csv').open()):
 k=tuple(x[z] for z in ['quantile','pilot','budget','predictor','stratum']);v=agg[k]
 for i,name in enumerate(['brier','log_loss']):
  mx=max(mx,abs(sum(z[i] for z in v)/len(v)-float(x[name])));count+=1
assert count==192 and mx<1e-12
# Check the probability-of-completion comparison using direct geometric summation.
q=.8;ta=3+math.log(4);tb=2*ta
assert (math.floor((10-ta)/4),math.floor((10-tb)/.25))==(1,4)
assert abs(sum(q*(1-q)**i for i in range(4))-.9984)<1e-12
assert round(ta+4/q,2)==9.39 and round(tb+.25/q,2)==9.09
assert 4-.25>q*(tb-ta)
result=dict(score_values_checked=count,maximum_score_difference=mx,prediction_rows=len(rows),independent_aggregation='stdlib CSV, table -> source -> equal-source',worked_example='PASS')
(R/'independent_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
