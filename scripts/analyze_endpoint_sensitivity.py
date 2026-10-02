"""Post-result endpoint sensitivity on frozen paths. No new campaign runs.
Independent NumPy Newton solver reproduces original SciPy ridge predictions first.
See evidence/consequential_gap_repairs_2026-09-26_012131/ before-edit contract.
"""
from pathlib import Path
import csv, json, hashlib
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1]
R=P/'evidence/eleven_review_remediation_2026-09-21_001619/results'
O=P/'evidence/consequential_gap_repairs_2026-09-26_012131/results'; O.mkdir(exist_ok=True)
S=R.parent/'sources/gollum/data'

def read(n): return list(csv.DictReader((R/n).open()))
def write(n, rows):
 with (O/n).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
traces=json.loads((R/'finite_table_traces.json').read_text())
meta=json.loads((R/'finite_table_sources.json').read_text())
oldfeatures=read('finite_table_features_outcomes.csv')
sim={(r['family'],r['table'],int(r['seed']),int(r['pilot'])):float(r['kernel_similarity']) for r in oldfeatures}
threshold={}
for m in meta:
 path=S/m['family']/(m['table']+'.csv')
 assert hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256']
 data=list(csv.DictReader(path.open())); y=np.array([float(data[i]['objective']) for i in m['rows']])
 for q in [.90,.95,.99]: threshold[m['family'],m['table'],q]=float(np.quantile(y,q))
 assert abs(threshold[m['family'],m['table'],.9]-m['threshold'])<1e-8
families=sorted({m['family'] for m in meta})

def weights(rows):
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[r['family'],r['table']].append(i)
 fams=sorted({f for f,t in groups}); w=np.zeros(len(rows))
 for (f,t),inds in groups.items():w[inds]=len(rows)/(len(fams)*sum(ff==f for ff,tt in groups)*len(inds))
 return w

def sigmoid(z): return np.exp(-np.logaddexp(0,-z))
def fit(tr,te,cols):
 x=np.array([[r[c] for c in cols] for r in tr]); xt=np.array([[r[c] for c in cols] for r in te]);y=np.array([r['observed'] for r in tr])
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-10]=1
 x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);xt=np.column_stack([np.ones(len(xt)),(xt-mu)/sd]);w=weights(tr)
 b=np.zeros(x.shape[1]);pen=np.ones(len(b));pen[0]=0
 def obj(b):
  z=x@b;return (w*(np.logaddexp(0,z)-y*z)).sum()+.5*(pen*b*b).sum()
 for i in range(150):
  z=x@b;p=sigmoid(z);g=x.T@(w*(p-y))+pen*b
  if abs(g).max()<1e-8:break
  h=x.T@((w*p*(1-p))[:,None]*x)+np.diag(pen)
  step=np.linalg.solve(h,g); rate=1.; current=obj(b)
  while obj(b-rate*step)>current-1e-4*rate*(g@step):
   rate*=.5
   if rate<1e-10:break
  b-=rate*step
 assert abs(g).max()<1e-5, (abs(g).max(),i)
 return sigmoid(xt@b)

def family_scores(pred):
 table=defaultdict(list)
 for r in pred:
  for stratum in ['all','not_yet_hit']:
   if stratum=='not_yet_hit' and r['hit_already']:continue
   table[(r['quantile'],r['family'],r['table'],r['pilot'],r['budget'],r['predictor'],stratum)].append(r)
 out=[]
 for key,rows in table.items():
  q,f,t,p,b,name,st=key;y=np.array([r['observed'] for r in rows]);pr=np.clip([r['predicted'] for r in rows],1e-6,1-1e-6)
  out.append(dict(quantile=q,family=f,table=t,pilot=p,budget=b,predictor=name,stratum=st,campaigns=len(rows),brier=float(np.mean((pr-y)**2)),log_loss=float(np.mean(-y*np.log(pr)-(1-y)*np.log1p(-pr))),hit_rate=float(y.mean())))
 groups=defaultdict(list)
 for r in out:groups[tuple(r[k] for k in ['quantile','family','pilot','budget','predictor','stratum'])].append(r)
 fam=[]
 for key,rows in groups.items():
  d=dict(zip(['quantile','family','pilot','budget','predictor','stratum'],key))
  d.update({k:float(np.mean([r[k] for r in rows])) for k in ['brier','log_loss','hit_rate']});fam.append(d)
 return out,fam
pred=[]; checks=[]
cols_base=['log_pilot','log_budget'];cols_full=cols_base+['log_candidates','best_gap','mean_gap','dispersion','kernel_similarity']
for q in [.90,.95,.99]:
 rows=[]
 for trc in traces:
  f,t,seed=trc['family'],trc['table'],trc['seed'];values=np.array(trc['outcomes']);cut=threshold[f,t,q]
  for pilot in [5,10]:
   seen=values[:pilot];scale=max(abs(cut),float(seen.std()),1e-8)
   for budget in [20,40]:
    rows.append(dict(quantile=q,family=f,table=t,seed=seed,pilot=pilot,budget=budget,hit_already=bool((seen>=cut).any()),observed=int((values[:budget]>=cut).any()),log_pilot=np.log(pilot),log_budget=np.log(budget),log_candidates=np.log(512),best_gap=(seen.max()-cut)/scale,mean_gap=(seen.mean()-cut)/scale,dispersion=seen.std()/scale,kernel_similarity=sim[f,t,seed,pilot]))
 for f in families:
  tr=[r for r in rows if r['family']!=f and not r['hit_already']];te=[r for r in rows if r['family']==f]
  for name,cols in [('no_target',cols_base),('best',cols_base+['log_candidates','best_gap']),('pilot_summary',cols_full)]:
   probs=fit(tr,te,cols)
   for r,pr in zip(te,probs):pred.append({k:r[k] for k in ['quantile','family','table','seed','pilot','budget','hit_already','observed']}|dict(predictor=name,predicted=1. if r['hit_already'] else float(pr)))
   # Check test labels never enter fitting, with changed targets but same observed features.
   altered=[dict(r,observed=1-r['observed']) for r in te]
   assert np.array_equal(probs,fit(tr,altered,cols));checks.append([q,f,name,'test_label_invariance'])
  for r in te:
   eligible=[v for v in tr if v['pilot']==r['pilot'] and v['budget']==r['budget']];w=weights(eligible)
   pr=float(np.average([v['observed'] for v in eligible],weights=w))
   pred.append({k:r[k] for k in ['quantile','family','table','seed','pilot','budget','hit_already','observed']}|dict(predictor='prevalence',predicted=1. if r['hit_already'] else pr))
 print('quantile completed',q,flush=True)
old={(r['family'],r['table'],int(r['seed']),int(r['pilot']),int(r['budget']),r['predictor']):float(r['predicted']) for r in read('finite_table_predictions.csv')}
diffs=[abs(r['predicted']-old[r['family'],r['table'],r['seed'],r['pilot'],r['budget'],r['predictor']]) for r in pred if r['quantile']==.9 and r['predictor']!='prevalence']
assert max(diffs)<1e-5, max(diffs)
write('predictions.csv',pred);tab,fam=family_scores(pred);write('table_scores.csv',tab);write('family_scores.csv',fam)
keys=['quantile','pilot','budget','predictor','stratum']; group=defaultdict(list)
for r in fam:group[tuple(r[k] for k in keys)].append(r)
summary=[]
for key,rows in group.items():summary.append(dict(zip(keys,key))|{k:float(np.mean([r[k] for r in rows])) for k in ['brier','log_loss','hit_rate']})
write('summary.csv',summary)
ix={(r['quantile'],r['family'],r['pilot'],r['budget'],r['predictor'],r['stratum']):r for r in fam}
paired=[]
for q in [.9,.95,.99]:
 for pilot in [5,10]:
  for budget in [20,40]:
   for st in ['all','not_yet_hit']:
    ds=[ix[q,f,pilot,budget,'pilot_summary',st]['brier']-ix[q,f,pilot,budget,'no_target',st]['brier'] for f in families]
    for f,d in zip(families,ds):paired.append(dict(quantile=q,pilot=pilot,budget=budget,stratum=st,family=f,brier_delta=d,leave_one_source_mean=float(np.mean([v for ff,v in zip(families,ds) if ff!=f]))))
write('paired_family_differences.csv',paired)
(O/'checks.json').write_text(json.dumps(dict(original_predictions_checked=len(diffs),maximum_prediction_difference=max(diffs),label_invariance_checks=len(checks),traces=len(traces),prediction_records=len(pred),quantiles=[.9,.95,.99],source_hashes_verified=len(meta)),indent=2))
print(json.dumps(json.loads((O/'checks.json').read_text()),indent=2))
for r in summary:
 if r['pilot']==5 and r['budget']==20 and r['stratum']=='all':print(r)
