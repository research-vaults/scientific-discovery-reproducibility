"""Exploratory budget-specific forecasting diagnostic on frozen chemical paths.
Contract: evidence/three_review_remediation_2026-09-26_221254/ANALYSIS_CONTRACT.md
Writes only the new analysis results; historical inputs remain unchanged.
"""
from pathlib import Path
import csv, json, hashlib
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1]
R=P/'evidence/eleven_review_remediation_2026-09-21_001619/results'
O=P/'evidence/three_review_remediation_2026-09-26_221254/results'; O.mkdir(exist_ok=True)
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
def fit(tr,te,cols,standardization_rows=None,penalty_scale=1.):
 x=np.array([[r[c] for c in cols] for r in tr]); xt=np.array([[r[c] for c in cols] for r in te]);y=np.array([r['observed'] for r in tr])
 xs=np.array([[r[c] for c in cols] for r in (standardization_rows or tr)]);mu=xs.mean(0);sd=xs.std(0);sd[sd<1e-10]=1
 x=np.column_stack([np.ones(len(x)),(x-mu)/sd]);xt=np.column_stack([np.ones(len(xt)),(xt-mu)/sd]);w=weights(tr)
 b=np.zeros(x.shape[1]);pen=np.full(len(b),penalty_scale);pen[0]=0
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
  out.append(dict(quantile=q,family=f,table=t,pilot=p,budget=b,predictor=name,stratum=st,campaigns=len(rows),brier=float(np.mean((pr-y)**2)),log_loss=float(np.mean(-y*np.log(pr)-(1-y)*np.log1p(-pr))),hit_rate=float(y.mean()),mean_prediction=float(pr.mean()),bias_squared=float(np.mean(pr-y)**2),residual_variance=float(np.var(pr-y))))
 groups=defaultdict(list)
 for r in out:groups[tuple(r[k] for k in ['quantile','family','pilot','budget','predictor','stratum'])].append(r)
 fam=[]
 for key,rows in groups.items():
  d=dict(zip(['quantile','family','pilot','budget','predictor','stratum'],key))
  d.update({k:float(np.mean([r[k] for r in rows])) for k in ['brier','log_loss','hit_rate','mean_prediction','bias_squared','residual_variance']});fam.append(d)
 return out,fam

pred=[];invariance=0; all_rows=[]
cols_base=['log_pilot','log_budget'];cols_full=cols_base+['log_candidates','best_gap','mean_gap','dispersion','kernel_similarity']
for q in [.90,.95,.99]:
 rows=[]
 for trc in traces:
  f,t,seed=trc['family'],trc['table'],trc['seed'];values=np.array(trc['outcomes']);cut=threshold[f,t,q]
  for pilot in [5,10]:
   seen=values[:pilot];scale=max(abs(cut),float(seen.std()),1e-8)
   for budget in [20,40]:
    rows.append(dict(quantile=q,family=f,table=t,seed=seed,pilot=pilot,budget=budget,hit_already=bool((seen>=cut).any()),observed=int((values[:budget]>=cut).any()),log_pilot=np.log(pilot),log_budget=np.log(budget),log_candidates=np.log(512),best_gap=(seen.max()-cut)/scale,mean_gap=(seen.mean()-cut)/scale,dispersion=seen.std()/scale,kernel_similarity=sim[f,t,seed,pilot]))

 all_rows.extend(rows)
 for f in families:
  tr=[r for r in rows if r['family']!=f and not r['hit_already']];te=[r for r in rows if r['family']==f]
  for name,cols in [('no_target',cols_base),('best',cols_base+['log_candidates','best_gap']),('pilot_summary',cols_full)]:
   for variant in ['pooled','separate','separate_matched_penalty']:
    batches=[(tr,te,1.)] if variant=='pooled' else [( [r for r in tr if r['budget']==h],[r for r in te if r['budget']==h], (sum(r['budget']==h for r in tr)/len(tr) if variant=='separate_matched_penalty' else 1.)) for h in [20,40]]
    for train,test,pen in batches:
     probs=fit(train,test,cols,tr,pen)
     altered=[dict(r,observed=1-r['observed']) for r in test]
     assert np.array_equal(probs,fit(train,altered,cols,tr,pen));invariance+=1
     for r,pr in zip(test,probs):pred.append({k:r[k] for k in ['quantile','family','table','seed','pilot','budget','hit_already','observed']}|dict(predictor=name+'__'+variant,predicted=1. if r['hit_already'] else float(pr)))
 print('Completed',q,flush=True)
old=list(csv.DictReader((P/'evidence/consequential_gap_repairs_2026-09-26_012131/results/predictions.csv').open()))
ix={(float(r['quantile']),r['family'],r['table'],int(r['seed']),int(r['pilot']),int(r['budget']),r['predictor']):float(r['predicted']) for r in old}
diffs=[abs(r['predicted']-ix[r['quantile'],r['family'],r['table'],r['seed'],r['pilot'],r['budget'],r['predictor'].split('__')[0]]) for r in pred if r['predictor'].endswith('__pooled')]
assert max(diffs)<1e-10,max(diffs)
write('features_outcomes.csv',all_rows);write('predictions.csv',pred);tab,fam=family_scores(pred);write('table_scores.csv',tab);write('family_scores.csv',fam)
keys=['quantile','pilot','budget','predictor','stratum'];group=defaultdict(list)
for r in fam:group[tuple(r[k] for k in keys)].append(r)
summary=[]
for key,rows in group.items():summary.append(dict(zip(keys,key))|{k:float(np.mean([r[k] for r in rows])) for k in ['brier','log_loss','hit_rate','mean_prediction','bias_squared','residual_variance']})
write('summary.csv',summary)
assert max(abs(r['brier']-r['bias_squared']-r['residual_variance']) for r in tab)<1e-12
pairs=defaultdict(dict)
for r in pred:pairs[tuple(r[k] for k in ['quantile','family','table','seed','pilot','predictor'])][r['budget']]=r
mon=[]
for key,ds in pairs.items():
 assert ds[40]['observed']>=ds[20]['observed']
 mon.append(dict(zip(['quantile','family','table','seed','pilot','predictor'],key))|dict(p20=ds[20]['predicted'],p40=ds[40]['predicted'],violation=ds[40]['predicted']<ds[20]['predicted']-1e-12))
write('horizon_pairs.csv',mon)
by=defaultdict(list)
for r in mon:by[r['quantile'],r['pilot'],r['predictor']].append(r)
write('horizon_summary.csv',[dict(quantile=k[0],pilot=k[1],predictor=k[2],pairs=len(v),violations=sum(r['violation'] for r in v)) for k,v in by.items()])
checks=dict(source_hashes_verified=len(meta),original_predictions_checked=len(diffs),maximum_prediction_difference=max(diffs),test_label_invariance_checks=invariance,rows=len(all_rows),prediction_records=len(pred),nested_outcome_pairs_checked=len(mon),brier_decomposition_verified=True)
(O/'checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
for r in summary:
 if r['quantile']==.99 and r['pilot']==5 and r['stratum']=='all':print(r)
