"""Review-driven descriptor checks; frozen plan in eleven-review evidence directory."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from scipy.stats import norm
P=Path(__file__).resolve().parents[1]
E=P/'evidence/eleven_review_remediation_2026-09-21_001619'; O=E/'results';O.mkdir(parents=True,exist_ok=True)
S=P/'evidence/review_remediation_2026-09-19/sources/ObsScaling/eval_results'
tree=ast.parse((P/'scripts/analyze_review_evidence.py').read_text())
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'pcs','regress','metrics'}],type_ignores=[]),'existing_helpers','exec'))
src=['MMLU','ARC-C','HellaSwag','Winograd','TruthfulQA','GSM8K','XWinograd']
base=pd.read_csv(S/'base_llm_benchmark_eval.csv').dropna(subset=src+['Model Family'])
post=pd.read_csv(S/'base_llm_post_training_eval.csv');frames=[]
for task in ['logical_deduction','tracking_shuffled_objects']:
 for n,word in [(3,'three'),(5,'five'),(7,'seven')]:
  col=f'bbh_cot_fewshot_{task}_{word}_objects_3_exact_match,strict-match'
  m=base.merge(post[['Model',col]],on='Model',validate='one_to_one').dropna(subset=[col])
  frames.append(m[['Model','Model Family']+src].assign(task_family=task,n=n,y=m[col]))
a=pd.concat(frames,ignore_index=True);assert len(a)==438 and a.Model.nunique()==73
families=sorted(a['Model Family'].unique());tasks=sorted(a.task_family.unique())
X=a[src].to_numpy();Y=a.y.to_numpy().copy();C=np.log(a.n.to_numpy());F=a['Model Family'].to_numpy();T=a.task_family.to_numpy();NS=a.n.to_numpy()

def predict(train,test,multiplicity=None,small=False):
 # PCA counts each unique training model once PER sampled family occurrence.
 unique=a.iloc[train].drop_duplicates('Model').index.to_numpy()
 if multiplicity is not None:
  unique=np.repeat(unique,[multiplicity[families.index(F[i])] for i in unique])
 _,x=pcs(X[unique],X[train]);_,xt=pcs(X[unique],X[test])
 c=(C[train]-C[train].mean())/C[train].std();ct=(C[test]-C[train].mean())/C[train].std()
 add=np.column_stack([x,c]);addt=np.column_stack([xt,ct])
 out={}
 for name,u,v in [('PC3',x,xt),('additive',add,addt)]:
  out[name]=regress(u,Y[train],v,penalty=1)[0]
 if small:return out
 quad=np.column_stack([x]+[x[:,i]*x[:,j] for i in range(3) for j in range(i,3)])
 quadt=np.column_stack([xt]+[xt[:,i]*xt[:,j] for i in range(3) for j in range(i,3)])
 out['quadratic_PC3']=regress(quad,Y[train],quadt,penalty=1)[0]
 for link,clip in [('probit',.005),('logit',.001),('logit',.01)]:
  for name,u,v in [('PC3',x,xt),('additive',add,addt)]:out[f'{name}_{link}_{clip}']=regress(u,Y[train],v,penalty=1,link=link,clip=clip)[0]
 for name,u,v in [('PC3',x,xt),('additive',add,addt)]:
  z=np.column_stack([np.ones(len(u)),u]);zt=np.column_stack([np.ones(len(v)),v]);b=np.linalg.solve(z.T@z+np.diag([0]+[1]*u.shape[1]),z.T@Y[train]);out[name+'_rate']=np.clip(zt@b,0,1)
 floor=1/NS[train];ft=1/NS[test];rel=(Y[train]-floor)/(1-floor)
 out['chance_floor']=ft+(1-ft)*regress(x,rel,xt,penalty=1)[0]
 out['chance_plus_count']=ft+(1-ft)*regress(add,rel,addt,penalty=1)[0]
 # Same information budget, no forced lower floor: chance enters as a feature.
 ch=np.column_stack([x,1/NS[train],c]);cht=np.column_stack([xt,1/NS[test],ct])
 out['chance_feature_plus_count']=regress(ch,Y[train],cht,penalty=1)[0]
 return out

def folds(axis,counts=None):
 fs=families if counts is None else [f for f,w in zip(families,counts) if w]
 pairs=([(None,t) for t in tasks] if axis=='task_family' else [(f,None) for f in fs] if axis=='model_family' else [(None,None)] if axis=='larger' else [(f,t) for f in fs for t in tasks])
 for f,t in pairs:
  tr=np.ones(len(a),bool);te=np.ones(len(a),bool)
  if f is not None:tr&=F!=f;te&=F==f
  if t is not None:tr&=T!=t;te&=T==t
  if axis in ['larger','joint_families_larger']:tr&=NS<7;te&=NS==7
  tri=np.flatnonzero(tr);tei=np.flatnonzero(te)
  if counts is not None:
   tri=np.repeat(tri,[counts[families.index(F[i])] for i in tri]);tei=np.repeat(tei,[counts[families.index(F[i])] for i in tei])
  yield f,t,tri,tei

rows=[];fl=[];checks=0
for axis in ['task_family','model_family','larger','joint_families','joint_families_larger']:
 for f,t,tr,te in folds(axis):
  pp=predict(tr,te);keep=Y[te].copy();Y[te]=1-keep;check=predict(tr,te);Y[te]=keep
  assert not set(tr)&set(te)
  for name,vals in pp.items():
   assert np.array_equal(vals,check[name]);checks+=1
   for i,v in zip(te,vals):rows.append(dict(axis=axis,predictor=name,model=a.Model.iloc[i],model_family=F[i],task_family=T[i],objects=int(NS[i]),observed=float(Y[i]),predicted=float(v)))
  fl.append(dict(axis=axis,family=f,task=t,train_rows=len(tr),test_rows=len(te)))
 print('sensitivity',axis,flush=True)
df=pd.DataFrame(rows);df.to_csv(O/'descriptor_sensitivity_predictions.csv',index=False)
scores=[]
for (axis,name),g in df.groupby(['axis','predictor']):
 for task in ['all']+tasks:
  h=g if task=='all' else g[g.task_family==task]
  fam=[metrics(q.observed.to_numpy(),q.predicted.to_numpy()) for _,q in h.groupby('model_family')]
  scores.append(dict(axis=axis,predictor=name,task_family=task,cells=len(h),equal_cell=metrics(h.observed.to_numpy(),h.predicted.to_numpy()),equal_family={k:float(np.mean([d[k] for d in fam])) for k in fam[0]}))
# Full cluster bootstrap of the fitted cross-validation statistic; fixed task identities.
rng=np.random.default_rng(20260921);rep=[];weights=[]
for b in range(500):
 count=np.bincount(rng.integers(0,len(families),size=len(families)),minlength=len(families));weights.append(count.tolist())
 assert (count>0).sum()>3
 for axis in ['joint_families','joint_families_larger']:
  errors=[]
  for f,t,tr,te in folds(axis,count):
   pp=predict(tr,te,count,small=True)
   d=(Y[te]-pp['additive'])**2-(Y[te]-pp['PC3'])**2
   errors.extend(zip(F[te],T[te],d))
  er=pd.DataFrame(errors,columns=['family','task','delta'])
  for task in ['all']+tasks:
   h=er if task=='all' else er[er.task==task]
   # Equal-family draws: mean within each original family, then bootstrap multiplicity.
   fm=h.groupby('family').delta.mean()
   ef=np.average(fm.to_numpy(),weights=[count[families.index(f)] for f in fm.index])
   rep.append(dict(replicate=b,axis=axis,task_family=task,equal_cell_delta=float(h.delta.mean()),equal_family_delta=float(ef),distinct_families=int((count>0).sum())))
 if (b+1)%50==0:print('full refits',b+1,flush=True)
r=pd.DataFrame(rep);r.to_csv(O/'descriptor_refit_replicates.csv',index=False)
summary=[]
for (axis,task),g in r.groupby(['axis','task_family']):
 for metric in ['equal_cell_delta','equal_family_delta']:
  summary.append(dict(axis=axis,task_family=task,metric=metric,percentiles_025_50_975=np.quantile(g[metric],[.025,.5,.975]).tolist(),fraction_negative=float((g[metric]<0).mean())))
for filename,v in [('descriptor_sensitivity_scores.json',scores),('descriptor_refit_summary.json',summary),('descriptor_refit_weights.json',{'families':families,'weights':weights}),('descriptor_sensitivity_checks.json',{'label_invariance_checks':checks,'folds':fl,'below_chance_cells':int((Y<1/NS).sum()),'cells':len(a),'below_chance_by_task':{t:int(((Y<1/NS)&(T==t)).sum()) for t in tasks}})]:
 (O/filename).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
print('descriptor robustness complete',flush=True)
