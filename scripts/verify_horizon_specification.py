"""Independent strict-endpoint optimizer and reporting checks (no campaign runs)."""
from pathlib import Path
import csv,json,hashlib
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
P=Path(__file__).resolve().parents[1]; E=P/'evidence/three_review_remediation_2026-09-26_221254'; R=E/'results'
rows=list(csv.DictReader((R/'features_outcomes.csv').open()))
for r in rows:
 for k in ['quantile','log_pilot','log_budget','log_candidates','best_gap','mean_gap','dispersion','kernel_similarity']:r[k]=float(r[k])
 for k in ['observed','pilot','budget','seed']:r[k]=int(r[k])
 r['hit_already']=r['hit_already']=='True'
pred=list(csv.DictReader((R/'predictions.csv').open()));ix={(float(r['quantile']),r['family'],r['table'],int(r['seed']),int(r['pilot']),int(r['budget']),r['predictor']):float(r['predicted']) for r in pred}
rows=[r for r in rows if r['quantile']==.99]; diffs=[]; fits=0
for f in sorted({r['family'] for r in rows}):
 tr0=[r for r in rows if r['family']!=f and not r['hit_already']]
 for name,cols in [('no_target',['log_pilot','log_budget']),('pilot_summary',['log_pilot','log_budget','log_candidates','best_gap','mean_gap','dispersion','kernel_similarity'])]:
  train_features=np.array([[r[k] for k in cols] for r in tr0]);mu=train_features.mean(0);sd=train_features.std(0);sd[sd<1e-10]=1
  for variant in ['pooled','separate','separate_matched_penalty']:
   for budget in ([None] if variant=='pooled' else [20,40]):
    tr=[r for r in tr0 if budget is None or r['budget']==budget];te=[r for r in rows if r['family']==f and (budget is None or r['budget']==budget)]
    groups=defaultdict(list)
    for i,r in enumerate(tr):groups[r['family'],r['table']].append(i)
    w=np.zeros(len(tr))
    for (fam,tab),inds in groups.items():w[inds]=len(tr)/(3*sum(a==fam for a,b in groups)*len(inds))
    x=np.c_[np.ones(len(tr)),(np.array([[r[k] for k in cols] for r in tr])-mu)/sd];y=np.array([r['observed'] for r in tr])
    pen=np.ones(x.shape[1])*(len(tr)/len(tr0) if variant=='separate_matched_penalty' else 1);pen[0]=0
    def fun(b):
     z=x@b
     return np.dot(w,np.logaddexp(0,z)-y*z)+.5*np.dot(pen*b,b), x.T@(w*(expit(z)-y))+pen*b
    res=minimize(fun,np.zeros(x.shape[1]),jac=True,method='BFGS',options={'gtol':1e-8,'maxiter':2000})
    assert np.max(np.abs(fun(res.x)[1]))<1e-4
    xt=np.c_[np.ones(len(te)),(np.array([[r[k] for k in cols] for r in te])-mu)/sd];pp=expit(xt@res.x)
    for r,v in zip(te,pp):diffs.append(abs((1 if r['hit_already'] else v)-ix[.99,f,r['table'],r['seed'],r['pilot'],r['budget'],name+'__'+variant]))
    fits+=1
assert max(diffs)<1e-5,max(diffs)
# Historical review integrity is checked only in the internal archive.
summary=list(csv.DictReader((R/'summary.csv').open()))
assert len(summary)==3*2*2*9*2
# Independently reconstruct every all-campaign mean from raw predictions.
by=defaultdict(list)
for r in pred:by[float(r['quantile']),int(r['pilot']),int(r['budget']),r['predictor'],r['family'],r['table']].append(r)
sc=defaultdict(list)
for key,rs in by.items():sc[key[:-1]].append(np.mean([(np.clip(float(r['predicted']),1e-6,1-1e-6)-int(r['observed']))**2 for r in rs]))
ss=defaultdict(list)
for key,vs in sc.items():ss[key[:-1]].append(np.mean(vs))
for r in summary:
 if r['stratum']=='all':assert abs(np.mean(ss[float(r['quantile']),int(r['pilot']),int(r['budget']),r['predictor']])-float(r['brier']))<1e-12
out=dict(independent_scipy_fits=fits,predictions_checked=len(diffs),maximum_prediction_difference=max(diffs),all_campaign_brier_summaries_reconstructed=len(ss))
(E/'independent_checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
