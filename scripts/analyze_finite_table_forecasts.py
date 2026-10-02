"""Outcome-reveal-only finite experimental table replay; see locked contract."""
from pathlib import Path
import json,hashlib,math
import numpy as np
import pandas as pd
from scipy.special import expit,gammaln
from scipy.optimize import minimize
P=Path(__file__).resolve().parents[1];E=P/'evidence/eleven_review_remediation_2026-09-21_001619';O=E/'results';S=E/'sources/gollum/data'
FAMILIES=['additives','buchwald-hartwig','c2-yield','suzuki-miyaura'];EXCLUDE={'objective','rxn','procedure','product','default_features','name','crystal_score'}

def load(path):
 d=pd.read_csv(path);cols=[c for c in d.columns if c not in EXCLUDE]
 d=d.dropna(subset=cols+['objective']).drop_duplicates(cols,keep='first')
 if len(d)>512:
  rng=np.random.default_rng(20260921);d=d.iloc[np.sort(rng.choice(len(d),512,replace=False))]
 assert len(d)>=40 and cols
 # Endpoint percentile is fixed from the admitted candidate table, not the run.
 y=d.objective.to_numpy();threshold=float(np.quantile(y,.9));dist=np.zeros((len(d),len(d)))
 for c in cols:
  if pd.api.types.is_numeric_dtype(d[c]):
   x=d[c].to_numpy();scale=np.ptp(x)
   if scale:dist+=((x[:,None]-x[None,:])/scale)**2
  else:
   x=d[c].astype(str).to_numpy();dist+=(x[:,None]!=x[None,:])
 K=np.exp(-2*dist/len(cols))
 return d,K,y,threshold,cols

def next_index(K,ids,values):
 n=len(K);ii=np.array(ids);vals=np.asarray(values);sd=vals.std();z=(vals-vals.mean())/(sd if sd>1e-8 else 1.)
 C=K[np.ix_(ii,ii)]+.01*np.eye(len(ii));cross=K[:,ii]
 sol=np.linalg.solve(C,np.column_stack([z,cross.T]));mu=cross@sol[:,0]
 var=np.maximum(0,1-np.sum(cross*sol[:,1:].T,axis=1));acq=mu+np.sqrt(var);acq[ii]=-np.inf
 return int(np.argmax(acq))

def run(K,oracle,seed,limit=40):
 ids=list(np.random.default_rng(seed).choice(len(K),5,replace=False));values=[float(oracle[i]) for i in ids]
 while len(ids)<limit:
  j=next_index(K,ids,values);assert j not in ids;ids.append(j);values.append(float(oracle[j]))
 return ids,values

records=[];traces=[];meta=[];prefix_checks=0
for fam in FAMILIES:
 for path in sorted((S/fam).glob('*.csv')):
  d,K,y,threshold,cols=load(path);table=path.stem
  meta.append(dict(family=fam,table=table,candidates=len(y),input_columns=cols,threshold=threshold,hit_fraction=float(np.mean(y>=threshold)),rows=d.index.tolist(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  for seed in range(50):
   ids,values=run(K,y,20260921+seed);assert len(set(ids))==40
   if seed==0:
    # Future oracle changes cannot change proposals through the observed prefix.
    yy=y.copy();rest=np.setdiff1d(np.arange(len(y)),ids[:10]);yy[rest]=yy[rest[::-1]]
    ii,vv=run(K,yy,20260921,limit=10);assert ii==ids[:10] and vv==values[:10];prefix_checks+=1
   traces.append(dict(family=fam,table=table,seed=seed,candidate_rows=[int(d.index[i]) for i in ids],outcomes=values))
   for pilot in [5,10]:
    seen=np.array(values[:pilot]);scale=max(abs(threshold),float(seen.std()),1e-8);hit=bool((seen>=threshold).any())
    sub=K[np.ix_(ids[:pilot],ids[:pilot])];sim=float((sub.sum()-pilot)/(pilot*(pilot-1)))
    for budget in [20,40]:
     records.append(dict(family=fam,table=table,seed=seed,pilot=pilot,budget=budget,candidates=len(y),hit_already=hit,eventual_hit=bool((np.array(values[:budget])>=threshold).any()),best_gap=float((seen.max()-threshold)/scale),mean_gap=float((seen.mean()-threshold)/scale),dispersion=float(seen.std()/scale),kernel_similarity=sim))
  print('finite-table replay',fam,table,flush=True)
# Oracle-informed uniform-search probability, not a forecast of the GP policy.
# Uses full-table hit counts solely for the contract's reference calculation.
reference=[]
for m in meta:
 n=m['candidates'];h=round(n*m['hit_fraction'])
 for b in [20,40]:
  prob=1-math.comb(n-h,b)/math.comb(n,b)
  reference.append(dict(family=m['family'],table=m['table'],candidates=n,archive_hits=h,budget=b,oracle_uniform_hit_probability=prob))
pd.DataFrame(reference).to_csv(O/'finite_table_oracle_random_reference.csv',index=False)
df=pd.DataFrame(records);df.to_csv(O/'finite_table_features_outcomes.csv',index=False)
(O/'finite_table_traces.json').write_text(json.dumps(traces,indent=2)+'\n');(O/'finite_table_sources.json').write_text(json.dumps(meta,indent=2)+'\n')

def fit(tr,te,cols):
 x=tr[cols].to_numpy(float);xt=te[cols].to_numpy(float);y=tr.eventual_hit.to_numpy(float)
 mu=x.mean(0);sd=x.std(0);sd[sd<1e-10]=1;x=(x-mu)/sd;xt=(xt-mu)/sd
 x=np.column_stack([np.ones(len(x)),x]);xt=np.column_stack([np.ones(len(xt)),xt])
 # Balance training source families, then tables, so large archives do not dominate.
 counts=tr.groupby(['family','table']).size();nf=tr.family.nunique();nt=tr.groupby('family').table.nunique()
 w=np.array([len(tr)/(nf*nt[r.family]*counts[(r.family,r.table)]) for r in tr.itertuples()])
 def objective(b):
  z=x@b;p=expit(z);loss=np.sum(w*(np.logaddexp(0,z)-y*z))+.5*np.sum(b[1:]**2)
  g=x.T@(w*(p-y));g[1:]+=b[1:];return loss,g
 res=minimize(objective,np.zeros(x.shape[1]),jac=True,method='BFGS',options={'gtol':1e-7,'maxiter':500})
 assert np.linalg.norm(res.jac)<1e-4
 return expit(xt@res.x)
for c in ['pilot','budget','candidates']:df['log_'+c]=np.log(df[c])
models={'no_target':['log_pilot','log_budget'],'count':['log_pilot','log_budget','log_candidates'],'best':['log_pilot','log_budget','log_candidates','best_gap'],'pilot_summary':['log_pilot','log_budget','log_candidates','best_gap','mean_gap','dispersion','kernel_similarity']}
pred=[]
for fam in FAMILIES:
 tr=df[(df.family!=fam)&(~df.hit_already)].copy();te=df[df.family==fam].copy()
 assert not set(tr.family)&set(te.family)
 for name,cols in models.items():
  p=fit(tr,te,cols);p[te.hit_already.to_numpy()]=1
  for row,pp in zip(te.itertuples(),p):pred.append(dict(family=fam,table=row.table,seed=row.seed,pilot=row.pilot,budget=row.budget,predictor=name,hit_already=row.hit_already,observed=int(row.eventual_hit),predicted=float(pp)))
pd.DataFrame(pred).to_csv(O/'finite_table_predictions.csv',index=False)
x=pd.DataFrame(pred);scores=[];cal=[]
for (fam,table,pilot,budget,name),g in x.groupby(['family','table','pilot','budget','predictor']):
 for stratum in ['all','not_yet_hit']:
  h=g if stratum=='all' else g[~g.hit_already]
  if not len(h):continue
  p=np.clip(h.predicted.to_numpy(),1e-6,1-1e-6);y=h.observed.to_numpy()
  scores.append(dict(family=fam,table=table,pilot=int(pilot),budget=int(budget),predictor=name,stratum=stratum,campaigns=len(h),brier=float(np.mean((p-y)**2)),log_loss=float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))),hit_rate=float(y.mean())))
for (pilot,budget,name),g in x.groupby(['pilot','budget','predictor']):
 for lo in np.arange(0,1,.2):
  h=g[(g.predicted>=lo)&(g.predicted<lo+.2+ (1e-8 if lo>.79 else 0))]
  if len(h):cal.append(dict(pilot=int(pilot),budget=int(budget),predictor=name,bin_lower=float(lo),n=len(h),mean_prediction=float(h.predicted.mean()),observed_rate=float(h.observed.mean())))
s=pd.DataFrame(scores);family=s.groupby(['family','pilot','budget','predictor','stratum'])[['brier','log_loss','hit_rate']].mean().reset_index();summary=family.groupby(['pilot','budget','predictor','stratum'])[['brier','log_loss','hit_rate']].mean().reset_index()
s.to_csv(O/'finite_table_scores.csv',index=False);family.to_csv(O/'finite_table_family_scores.csv',index=False);summary.to_csv(O/'finite_table_summary.csv',index=False)
(O/'finite_table_checks.json').write_text(json.dumps(dict(tables=len(meta),source_families=len(FAMILIES),campaigns=len(traces),prefix_invariance_checks=prefix_checks,selection_cost_per_observation=1,forecast_rows=len(df),prediction_rows=len(pred),calibration=cal),indent=2)+'\n')
print(summary.to_string(index=False),flush=True)
