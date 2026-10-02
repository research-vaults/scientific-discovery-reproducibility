"""Independent arithmetic and source checks for the review-driven additions."""
from pathlib import Path
import csv,json,math,hashlib
from collections import defaultdict
from statistics import mean
P=Path(__file__).resolve().parents[1];E=P/'evidence/eleven_review_remediation_2026-09-21_001619';O=E/'results'
def read(n):return json.loads((O/n).read_text())
def rows(n):return list(csv.DictReader((O/n).open()))
def qtile(xs,p):
 v=sorted(xs);i=(len(v)-1)*p;j=int(i);return v[j]+(i-j)*(v[min(j+1,len(v)-1)]-v[j])
checks=0;maximum=0.
def same(x,y,tol=2e-12):
 global checks,maximum
 err=abs(x-y);maximum=max(maximum,err);assert err<tol,(x,y,err);checks+=1
pred=rows('descriptor_sensitivity_predictions.csv');groups=defaultdict(list)
for r in pred:groups[(r['axis'],r['predictor'])].append(r)
for s in read('descriptor_sensitivity_scores.json'):
 g=groups[(s['axis'],s['predictor'])];g=[r for r in g if s['task_family']=='all' or r['task_family']==s['task_family']]
 family=defaultdict(list);sq=[];ce=[]
 for r in g:
  y=float(r['observed']);p=min(1-1e-8,max(1e-8,float(r['predicted'])));v=((y-p)**2,-y*math.log(p)-(1-y)*math.log1p(-p));sq.append(v[0]);ce.append(v[1]);family[r['model_family']].append(v)
 same(mean(sq),s['equal_cell']['aggregate_rate_mse']);same(mean(ce),s['equal_cell']['cross_entropy'])
 for k,metric in enumerate(['aggregate_rate_mse','cross_entropy']):same(mean(mean(v[k] for v in f) for f in family.values()),s['equal_family'][metric])
# Reproduce the original predictor, not merely the newly emitted summaries.
old=list(csv.DictReader((P/'evidence/review_remediation_2026-09-19/results/descriptor_predictions.csv').open()))+list(csv.DictReader((P/'evidence/joint_holdout_2026-09-19_151855/results/predictions.csv').open()))
axmap={'target_family':'task_family','larger_target':'larger'};modelmap={'PC3':'PC3','PC3_count_additive':'additive','PC3_chance_floor':'chance_floor'}
index={(r['axis'],r['predictor'],r['model'],r['task_family'],r['objects']):r for r in pred}
for r in old:
 if r['predictor'] in modelmap:
  k=(axmap.get(r['axis'],r['axis']),modelmap[r['predictor']],r['model'],r['task_family'],r['objects']);same(float(r['predicted']),float(index[k]['predicted']))
rep=rows('descriptor_refit_replicates.csv');assert len(rep)==3000
for s in read('descriptor_refit_summary.json'):
 v=[float(r[s['metric']]) for r in rep if r['axis']==s['axis'] and r['task_family']==s['task_family']]
 for pp,w in zip([.025,.5,.975],s['percentiles_025_50_975']):same(qtile(v,pp),w)
w=read('descriptor_refit_weights.json');assert len(w['weights'])==500 and all(sum(x)==20 for x in w['weights'])
# Directly recompute posterior actions and conditional future reliability.
ps=[.1,.2,.3]
for r in read('operational_policy_actions.json'):
 n,k=r['N'],r['successes'];likes=[math.comb(n,k)*p**k*(1-p)**(n-k)/3 for p in ps];post=[v/sum(likes) for v in likes]
 for x,y in zip(post,r['posterior']):same(x,y)
 cap=1
 while sum(w*(1-p)**cap for w,p in zip(post,ps))>.1:cap+=1
 cand=11*cap<220;assert r['method']==('candidate' if cand else 'baseline') and r['cap']==(cap if cand else 22)
 success=sum(w*(1-(1-p)**cap) for w,p in zip(post,ps)) if cand else 1-.9**22
 same(success,r['posterior_success']);assert success>=.9-1e-12
for r in read('decision_robustness.json'):
 for method in ['directed','sequential']:
  d=r[method];same(d['adopt']+d['non_saving']+d['unresolved'],1);assert 0<=d['wrong']<=.05;assert d['pilot_cost']<=21*d['N_per_arm']+1e-9
# Finite-table labels must equal actual source measurements at retained row IDs.
metadata={r['table']:r for r in read('finite_table_sources.json')};source={}
for path in (E/'sources/gollum/data').rglob('*.csv'):
 if path.stem in metadata:source[path.stem]=list(csv.DictReader(path.open()))
trace={(r['table'],r['seed']):r for r in read('finite_table_traces.json')};assert len(trace)==550
for r in trace.values():
 assert len(r['candidate_rows'])==len(set(r['candidate_rows']))==40
 for i,v in zip(r['candidate_rows'],r['outcomes']):same(float(source[r['table']][i]['objective']),v,tol=1e-8)
features=rows('finite_table_features_outcomes.csv')
for r in features:
 t=trace[(r['table'],int(r['seed']))];n=int(r['pilot']);b=int(r['budget']);threshold=metadata[r['table']]['threshold']
 assert (r['hit_already']=='True')==any(x>=threshold for x in t['outcomes'][:n])
 assert (r['eventual_hit']=='True')==any(x>=threshold for x in t['outcomes'][:b])
# Uniform-search reference via sequential no-hit probabilities (independent formula).
for r in rows('finite_table_oracle_random_reference.csv'):
 n,h,b=int(r['candidates']),int(r['archive_hits']),int(r['budget'])
 same(1-math.prod((n-h-j)/(n-j) for j in range(b)),float(r['oracle_uniform_hit_probability']))
# Independently reconstruct table Brier scores from prediction rows.
pred=rows('finite_table_predictions.csv');groups=defaultdict(list)
for r in pred:groups[(r['family'],r['table'],r['pilot'],r['budget'],r['predictor'])].append(r)
for s in rows('finite_table_scores.csv'):
 g=groups[tuple(s[k] for k in ['family','table','pilot','budget','predictor'])]
 if s['stratum']=='not_yet_hit':g=[x for x in g if x['hit_already']=='False']
 same(mean((min(1-1e-6,max(1e-6,float(r['predicted'])))-int(r['observed']))**2 for r in g),float(s['brier']))
# Inputs and frozen plans must still have their locked identities.
for p,lock in [('FINITE_TABLE_CONTRACT.md','FINITE_TABLE_CONTRACT.lock.json')]:
 assert hashlib.sha256((E/p).read_bytes()).hexdigest()==json.loads((E/lock).read_text())['sha256']
for name,digest in json.loads((E/'sources/gollum_manifest.json').read_text())['files'].items():assert hashlib.sha256((E/name).read_bytes()).hexdigest()==digest
(O/'independent_checks.json').write_text(json.dumps({'status':'PASS','arithmetic_checks':checks,'maximum_absolute_difference':maximum,'original_predictors_reconstructed':True,'refit_replicates':500,'finite_table_traces':550,'scope':'Independent arithmetic and retained-source consistency; not an independent scientific review or new empirical measurements.'},indent=2)+'\n')
print('Independent review-remediation checks PASS',checks)
