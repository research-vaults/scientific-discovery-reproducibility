"""Independent reconstructions of load-bearing numerical claims and archive diagnostics."""
from pathlib import Path
import csv,json,itertools
import numpy as np
from scipy.special import logit,expit
from scipy.stats import binom,beta
P=Path(__file__).resolve().parents[1];E=P/'evidence/review_remediation_2026-09-19';O=E/'results'
def read(n):return json.loads((O/n).read_text())
checks=[]
# Reconstruct one held-out HumanEval family from covariance eigenvectors rather than SVD.
rows=list(csv.DictReader((E/'sources/ObsScaling/eval_results/base_llm_benchmark_eval.csv').open()))
cols=['MMLU','ARC-C','HellaSwag','Winograd','TruthfulQA','GSM8K','XWinograd','HumanEval']
rows=[r for r in rows if all(r[c] and r[c].lower()!='nan' for c in cols+['FLOPs (1E21)']) and float(r['FLOPs (1E21)'])>0]
family='Llama-2';tr=[r for r in rows if r['Model Family']!=family];te=[r for r in rows if r['Model Family']==family]
x=np.array([[float(r[c]) for c in cols[:-1]] for r in tr]);xt=np.array([[float(r[c]) for c in cols[:-1]] for r in te]);mu=x.mean(0);sd=x.std(0);z=(x-mu)/sd;zt=(xt-mu)/sd
vals,vec=np.linalg.eigh(z.T@z);vec=vec[:,np.argsort(vals)[::-1]];err=0
preds=list(csv.DictReader((O/'association_predictions.csv').open()))
for k in [1,3]:
 a=z@vec[:,:k];b=zt@vec[:,:k];sc=a.std(0);a=a/sc;b=b/sc
 design=np.column_stack([np.ones(len(a)),a]);y=logit(np.clip([float(r['HumanEval']) for r in tr],.005,.995));coef=np.linalg.lstsq(design,y,rcond=None)[0];pr=expit(np.column_stack([np.ones(len(b)),b])@coef)
 saved={r['model']:float(r['predicted']) for r in preds if r['target']=='HumanEval' and r['family']==family and r['predictor']==f'PC{k}'}
 for r,v in zip(te,pr):err=max(err,abs(saved[r['Model']]-v))
assert err<1e-12;checks.append({'test':'held-out HumanEval Llama-2 eigensystem reconstruction','max_abs_error':err,'rows':len(te)})
# Independent full-domain identity count for every hidden coefficient, not just selected example.
for n in range(2,9):
 x=np.arange(2**n);a=np.arange(1,2**n,2);out=(a[:,None]*x[None,:])%(2**n)
 match=np.all(out[:,None,:]==out[None,:,:],axis=2);assert np.array_equal(match,np.eye(len(a),dtype=bool))
checks.append({'test':'all hidden coefficients and candidate outputs','bit_widths':list(range(2,9)),'status':'PASS'})
# Direct product-binomial sum must match independent-pair dynamic programming.
n=100;x=np.arange(n+1);L=beta.ppf(.0125,x,n-x+1);U=beta.ppf(.9875,x+1,n-x);L[0]=0;U[-1]=1
with np.errstate(divide='ignore',invalid='ignore'):
 def B(p,c):return np.where(p<=0,np.inf,np.where(p>=1,c,c*np.ceil(np.log(.1)/np.log1p(-p))))
 adopt=B(L,11)[None,:]<B(U,10)[:,None];non=B(U,11)[None,:]>=B(L,10)[:,None]
 maxerr=0
 for r in read('pilot_extensions.json'):
  if r['design']=='fixed' and r['pair_dependence']=='independent':
   w=np.outer(binom.pmf(x,n,r['p0']),binom.pmf(x,n,r['p1']));maxerr=max(maxerr,abs(w[adopt].sum()-r['adopt']),abs(w[non].sum()-r['non_saving']))
assert maxerr<1e-12;checks.append({'test':'fixed-pilot product sum vs sequential-state implementation','max_abs_error':float(maxerr)})
# Reliability of all retained caps, including integer boundaries.
for r in read('synthesis.json'):
 assert r['cap_q90']/r['programs']>=.9 and (r['cap_q90']-1)/r['programs']<.9
for r in read('value_of_information.json'):
 assert r['posterior_optimal_expected_reserve']<=r['no_pilot_expected_reserve']+1e-9
 assert r['no_pilot_expected_reserve']-r['posterior_optimal_expected_reserve']<=r['perfect_information_upper_value']+1e-9
checks.append({'test':'reliability caps and information-value inequalities','status':'PASS'})
# Save calibration bins and subgroup losses; descriptive rate calibration, no item-level claims.
cal=[];groups=[]
for fname,keys in [('association_predictions.csv',['target','predictor']),('descriptor_predictions.csv',['axis','predictor'])]:
 rows=list(csv.DictReader((O/fname).open())); levels=sorted(set(tuple(r[k] for k in keys) for r in rows))
 for lev in levels:
  rr=[r for r in rows if tuple(r[k] for k in keys)==lev]
  for j in range(10):
   ss=[r for r in rr if min(int(float(r['predicted'])*10),9)==j]
   if ss:cal.append({'source':fname,**dict(zip(keys,lev)),'bin_left':j/10,'bin_right':(j+1)/10,'cells':len(ss),'mean_prediction':float(np.mean([float(r['predicted']) for r in ss])),'mean_observed_rate':float(np.mean([float(r['observed']) for r in ss]))})
  subgroup='family' if fname.startswith('association') else 'task_family'
  for sg in sorted(set(r[subgroup] for r in rr)):
   ss=[r for r in rr if r[subgroup]==sg];y=np.array([float(r['observed']) for r in ss]);p=np.array([float(r['predicted']) for r in ss]);groups.append({'source':fname,**dict(zip(keys,lev)),'subgroup':sg,'cells':len(ss),'rate_mse':float(np.mean((p-y)**2))})
(O/'calibration_bins.json').write_text(json.dumps(cal,indent=2));(O/'subgroup_scores.json').write_text(json.dumps(groups,indent=2));(O/'independent_checks.json').write_text(json.dumps({'status':'PASS','checks':checks},indent=2));print('Independent reconstructions PASS')
