"""Independent reconstruction of main-table losses and a single archived forecast.
No model calls, new endpoints or new datasets. Example selection is illustrative.
"""
from pathlib import Path
import csv, json, math
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1]
E=P/'evidence/maintrack_readability_2026-09-26_190243';E.mkdir(exist_ok=True)
S=P/'evidence/review_remediation_2026-09-19/sources/ObsScaling/eval_results'
cols=['MMLU','ARC-C','HellaSwag','Winograd','TruthfulQA','GSM8K','XWinograd']
base=list(csv.DictReader((S/'base_llm_benchmark_eval.csv').open()))
post={r['Model']:r for r in csv.DictReader((S/'base_llm_post_training_eval.csv').open())}
rows=[]
for task in ['logical_deduction','tracking_shuffled_objects']:
 for n,word in [(3,'three'),(5,'five'),(7,'seven')]:
  target=f'bbh_cot_fewshot_{task}_{word}_objects_3_exact_match,strict-match'
  for r in base:
   try: vals=[float(r[k]) for k in cols];y=float(post[r['Model']][target])
   except (ValueError,KeyError):continue
   if not r['Model Family'] or not all(math.isfinite(x) for x in vals+[y]):continue
   rows.append(dict(model=r['Model'],family=r['Model Family'],task=task,n=n,x=vals,y=y))
assert len(rows)==438
tr=[r for r in rows if r['family']!='Gemma' and r['task']!='logical_deduction' and r['n']<7]
te=next(r for r in rows if r['model']=='google/gemma-7b' and r['task']=='logical_deduction' and r['n']==7)
uniq=list({r['model']:r for r in tr}.values());u=np.array([r['x'] for r in uniq]);mu=u.mean(0);sd=u.std(0);sd=np.where(sd>1e-10,sd,1);z=(u-mu)/sd
_,_,vt=np.linalg.svd(z,full_matrices=False);v=vt[:3].T
if np.corrcoef(z@v[:,0],z.mean(1))[0,1]<0:v[:,0]*=-1
ps=(z@v).std(0);ps=np.where(ps>1e-10,ps,1)
x=((np.array([r['x'] for r in tr])-mu)/sd)@v/ps
xt=((np.array(te['x'])-mu)/sd)@v/ps
lc=np.log([r['n'] for r in tr]);center=lc.mean();scale=lc.std();count=(lc-center)/scale;test_count=(math.log(7)-center)/scale
y=np.clip([r['y'] for r in tr],.005,.995);yy=np.log(y/(1-y));records=[]
ret=list(csv.DictReader((P/'evidence/joint_holdout_2026-09-19_151855/results/predictions.csv').open()))
for name,features,test in [('PC3',x,xt),('PC3_count_additive',np.column_stack([x,count]),np.r_[xt,test_count])]:
 design=np.column_stack([np.ones(len(tr)),features]);test_design=np.r_[1.,test];coef=np.linalg.solve(design.T@design+np.diag([0]+[1]*features.shape[1]),design.T@yy);eta=float(test_design@coef);pred=1/(1+math.exp(-eta))
 r=next(r for r in ret if r['axis']=='joint_families_larger' and r['model']==te['model'] and r['task_family']==te['task'] and r['predictor']==name)
 assert abs(pred-float(r['predicted']))<1e-10
 records.append(dict(predictor=name,coefficients=coef.tolist(),design_row=test_design.tolist(),logit=eta,predicted=pred,observed=te['y'],squared_error=(pred-te['y'])**2,retained_prediction=float(r['predicted'])))
# Independently recompute every retained joint loss (including equal-family weighting).
groups=defaultdict(list)
for r in ret:
 for task in ['all',r['task_family']]:groups[r['axis'],r['predictor'],task].append(r)
scores=json.loads((P/'evidence/joint_holdout_2026-09-19_151855/results/scores.json').read_text());idx={(r['axis'],r['predictor'],r['task_family']):r for r in scores};checks=[]
for key,g in groups.items():
 by=defaultdict(list)
 for r in g:by[r['model_family']].append((float(r['predicted'])-float(r['observed']))**2)
 cell=sum(sum(t) for t in by.values())/len(g);family=sum(sum(t)/len(t) for t in by.values())/len(by)
 for weight,val in [('equal_cell',cell),('equal_family',family)]:
  saved=idx[key][weight]['aggregate_rate_mse'];assert abs(val-saved)<1e-12;checks.append(dict(axis=key[0],predictor=key[1],task=key[2],weighting=weight,mse=val,cells=len(g)))
res=dict(scope='Retrospective reconstruction, not independent scientific confirmation',example_selection='Named Gemma-7B seven-object deduction cell for explanation, not a representative effect estimate',training_cells=len(tr),training_models=len(uniq),training_families=len(set(r['family'] for r in tr)),training_tasks=sorted(set(r['task'] for r in tr)),training_object_counts=sorted(set(r['n'] for r in tr)),source_metrics=cols,source_scores=te['x'],source_center=mu.tolist(),source_scale=sd.tolist(),pca_loadings=v.tolist(),pc_scale=ps.tolist(),log_count_center=float(center),log_count_scale=float(scale),example=records,joint_mse_checks=checks)
(E/'reconstruction.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps({k:res[k] for k in ['training_cells','training_models','training_families','example']},indent=2));print('joint MSE checks:',len(checks))
# Main chemistry rows: reconstruct the nested average directly from saved campaigns.
C=P/'evidence/consequential_gap_repairs_2026-09-26_012131/results'
cp=list(csv.DictReader((C/'predictions.csv').open()));summ=list(csv.DictReader((C/'summary.csv').open()));ci={(float(r['quantile']),int(r['pilot']),int(r['budget']),r['predictor'],r['stratum']):r for r in summ}
chem=[];source_losses={}
for quantile in [.9,.95,.99]:
 for budget in [20,40]:
  for predictor in ['no_target','best','pilot_summary']:
   g=[r for r in cp if float(r['quantile'])==quantile and int(r['pilot'])==5 and int(r['budget'])==budget and r['predictor']==predictor]
   assert len(g)==550
   tables=defaultdict(list)
   for r in g:tables[r['family'],r['table']].append(((float(r['predicted'])-int(r['observed']))**2,int(r['observed'])))
   fs=defaultdict(list)
   for (family,table),values in tables.items():fs[family].append(np.mean(values,axis=0))
   av={family:np.mean(values,axis=0) for family,values in fs.items()};assert len(av)==4
   brier,hit=np.mean(list(av.values()),axis=0);ref=ci[quantile,5,budget,predictor,'all'];assert abs(brier-float(ref['brier']))<1e-12 and abs(hit-float(ref['hit_rate']))<1e-12
   chem.append(dict(quantile=quantile,budget=budget,predictor=predictor,brier=float(brier),hit_rate=float(hit),campaigns=len(g),origins=len(av)))
   for family,values in av.items():source_losses[quantile,budget,predictor,family]=float(values[0])
pairs=[]
for r in csv.DictReader((C/'paired_family_differences.csv').open()):
 if r['pilot']!='5' or r['budget']!='20' or r['stratum']!='all':continue
 q=float(r['quantile']);f=r['family'];d=source_losses[q,20,'pilot_summary',f]-source_losses[q,20,'no_target',f];assert abs(d-float(r['brier_delta']))<1e-12;pairs.append(dict(quantile=q,family=f,delta=d))
res['chemistry_main_checks']=chem;res['source_contrast_checks']=pairs
(E/'reconstruction.json').write_text(json.dumps(res,indent=2)+'\n')
print('Chemistry checks:',len(chem),'pooled rows and',len(pairs),'source contrasts')
