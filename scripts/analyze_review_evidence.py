"""Public archive diagnostics; specification: evidence/review_remediation_2026-09-19/ANALYSIS_CONTRACT.md."""
from pathlib import Path
import json, hashlib, csv
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42})
P=Path(__file__).resolve().parents[1]
E=P/'evidence/review_remediation_2026-09-19'
O=E/'results'; O.mkdir(exist_ok=True)
S=E/'sources/ObsScaling/eval_results'
METRICS=['MMLU','ARC-C','HellaSwag','Winograd','TruthfulQA','GSM8K','XWinograd','HumanEval']
D=pd.read_csv(S/'base_llm_benchmark_eval.csv')
RNG=np.random.default_rng(20260919)

def save(name,x):
 (O/name).write_text(json.dumps(x,indent=2,allow_nan=False))

def pcs(xtrain,xtest,k=3):
 mu=xtrain.mean(0); sd=xtrain.std(0); sd=np.where(sd>1e-10,sd,1)
 z=(xtrain-mu)/sd
 _,_,v=np.linalg.svd(z,full_matrices=False); v=v[:k].T
 if np.corrcoef(z@v[:,0],z.mean(1))[0,1]<0:v[:,0]*=-1
 a=z@v; b=((xtest-mu)/sd)@v
 ps=a.std(0); ps=np.where(ps>1e-10,ps,1)
 return a/ps,b/ps

def regress(x,y,xt,penalty=0,link='logit',clip=.005):
 trans=logit if link=='logit' else norm.ppf
 inv=expit if link=='logit' else norm.cdf
 a=np.column_stack([np.ones(len(x)),x]); b=np.column_stack([np.ones(len(xt)),xt])
 if penalty:
  reg=np.diag([0]+[penalty]*x.shape[1]); coef=np.linalg.solve(a.T@a+reg,a.T@trans(np.clip(y,clip,1-clip)))
 else:coef=np.linalg.lstsq(a,trans(np.clip(y,clip,1-clip)),rcond=None)[0]
 return inv(b@coef),coef

def metrics(y,p):
 p=np.clip(p,1e-8,1-1e-8)
 return {'aggregate_rate_mse':float(np.mean((y-p)**2)), 'cross_entropy':float(np.mean(-y*np.log(p)-(1-y)*np.log(1-p)))}

allpred=[]; summaries=[]
fig,axs=plt.subplots(1,2,figsize=(8.5,3.0))
for ti,target in enumerate(['HumanEval','GSM8K']):
 src=[c for c in METRICS if c!=target]
 d=D.dropna(subset=METRICS+['Model Family']).copy(); x=d[src].to_numpy(); y=d[target].to_numpy()
 assert np.all((x>=0)&(x<=1)) and np.all((y>=0)&(y<=1))
 z,_=pcs(x,x); pred,coef=regress(z[:,:1],y,z[:,:1])
 fam=d['Model Family'].to_numpy(); fs=np.unique(fam); boots=[]
 for it in range(2000):
  ix=np.concatenate([np.flatnonzero(fam==f) for f in RNG.choice(fs,len(fs),replace=True)])
  a,b=pcs(x[ix],x,1)
  # Each bootstrap axis is anchored by its spread across the original population.
  scale=b.std(); a=a/scale; b=b/scale
  _,c=regress(a,y[ix],a)
  center=b.mean(); intercept=c[0]+c[1]*center
  boots.append([float(intercept),float(c[1])])
 b=np.array(boots); pd.DataFrame(b,columns=['intercept_at_archive_mean','slope_per_archive_sd']).to_csv(O/f'{target}_bootstrap.csv',index=False)
 ci=np.quantile(b[:,1],[.025,.975]); thresholds=[]
 for q in [.2,.5]:
  point=(logit(q)-coef[0])/coef[1]; draws=(logit(q)-b[:,0])/b[:,1]
  positive=b[:,1]>0
  thresholds.append({'q':q,'source_sd_threshold':float(point),'bootstrap_percentiles_positive_slope':np.quantile(draws[positive],[.025,.975]).tolist(),'nonpositive_slopes':int((~positive).sum()),'outside_observed_source_range':bool(point<z[:,0].min() or point>z[:,0].max())})
 sensitivity=[]
 for clip in [.001,.005,.01]:
  _,c=regress(z[:,:1],y,z[:,:1],clip=clip);sensitivity.append({'link':'logit','clip':clip,'slope':float(c[1])})
 _,c=regress(z[:,:1],y,z[:,:1],link='probit');sensitivity.append({'link':'probit','clip':.005,'slope':float(c[1])})
 # Same population for all held-out comparisons.
 d=d[d['FLOPs (1E21)'].notna() & (d['FLOPs (1E21)']>0)].copy()
 for f in sorted(d['Model Family'].unique()):
  tr=d[d['Model Family']!=f]; te=d[d['Model Family']==f]
  a,bz=pcs(tr[src].to_numpy(),te[src].to_numpy()); yy=tr[target].to_numpy()
  forecasts={'target_mean':np.full(len(te),yy.mean())}
  for name,aa,bb in [('PC1',a[:,:1],bz[:,:1]),('PC3',a,bz),('log_compute',np.log(tr[['FLOPs (1E21)']].to_numpy()),np.log(te[['FLOPs (1E21)']].to_numpy()))]:
   forecasts[name]=regress(aa,yy,bb)[0]
  for name,pp in forecasts.items():
   for (_,r),p in zip(te.iterrows(),pp):allpred.append({'target':target,'family':f,'model':r['Model'],'predictor':name,'observed':r[target],'predicted':p})
 summary={'target':target,'models':len(x),'families':len(fs),'excluded_models':D.loc[~D['Model'].isin(D.dropna(subset=METRICS+['Model Family'])['Model']),'Model'].tolist(),'slope':float(coef[1]),'intercept':float(coef[0]),'slope_cluster_percentile95':ci.tolist(),'source_range': [float(z[:,0].min()),float(z[:,0].max())],'thresholds':thresholds,'sensitivities':sensitivity,'heldout_models':len(d),'heldout_families':int(d['Model Family'].nunique())}
 summaries.append(summary)
 axs[ti].scatter(z[:,0],y,s=15,alpha=.65,color='#245675'); xx=np.linspace(z[:,0].min(),z[:,0].max(),150)
 axs[ti].plot(xx,expit(coef[0]+coef[1]*xx),color='#c25436')
 curves=expit(np.array(boots)[:,0,None]+np.array(boots)[:,1,None]*xx)
 lo,hi=np.quantile(curves,[.025,.975],axis=0);axs[ti].fill_between(xx,lo,hi,color='#c25436',alpha=.18)
 axs[ti].set(title=f'{target}: {len(x)} models, {len(fs)} families',xlabel='Target-excluded source PC1 (SD)',ylabel='Published success rate')
 axs[ti].text(.04,.94,f'Slope {coef[1]:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]',transform=axs[ti].transAxes,va='top',fontsize=9)
 axs[ti].spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(P/'manuscript/figures/measured_coupling.pdf');plt.close(fig)
pd.DataFrame(allpred).to_csv(O/'association_predictions.csv',index=False)
for s in summaries:
 p=pd.DataFrame(allpred); p=p[p.target==s['target']]
 s['heldout_scores']={m:metrics(g.observed.to_numpy(),g.predicted.to_numpy()) for m,g in p.groupby('predictor')}
save('association_summary.json',summaries)

# Shared target-descriptor tests. No target outcomes enter PCA or metadata.
post=pd.read_csv(S/'base_llm_post_training_eval.csv'); src=METRICS[:-1]
base=D.dropna(subset=src+['Model Family']); rows=[]
for tf in ['logical_deduction','tracking_shuffled_objects']:
 for n,word in [(3,'three'),(5,'five'),(7,'seven')]:
  col=f'bbh_cot_fewshot_{tf}_{word}_objects_3_exact_match,strict-match' if tf=='logical_deduction' else f'bbh_cot_fewshot_{tf}_{word}_objects_3_exact_match,strict-match'
  # Both archive task names end with number_objects except tracking, which starts objects.
  if tf=='tracking_shuffled_objects':col=f'bbh_cot_fewshot_tracking_shuffled_objects_{word}_objects_3_exact_match,strict-match'
  assert col in post,col
  merged=base.merge(post[['Model',col]],on='Model').dropna(subset=[col])
  for _,r in merged.iterrows():rows.append({**{c:r[c] for c in ['Model','Model Family']+src},'task_family':tf,'n':n,'y':r[col]})
a=pd.DataFrame(rows);predrows=[]
folds=[('target_family',f,a.task_family==f) for f in sorted(a.task_family.unique())]+[('model_family',f,a['Model Family']==f) for f in sorted(a['Model Family'].unique())]+[('larger_target','7_objects',a.n==7)]
for axis,f,mask in folds:
 tr=a[~mask];te=a[mask]
 uniq=tr.drop_duplicates('Model'); mu=uniq[src].to_numpy()
 _,trainpc=pcs(mu,tr[src].to_numpy());_,testpc=pcs(mu,te[src].to_numpy())
 dn=np.log(tr.n.to_numpy());en=np.log(te.n.to_numpy()); center=dn.mean();scale=dn.std(); dn=(dn-center)/scale; en=(en-center)/scale
 models={'PC1':(trainpc[:,:1],testpc[:,:1]),'PC3':(trainpc,testpc),'PC3_count':(np.column_stack([trainpc,dn,trainpc*dn[:,None]]),np.column_stack([testpc,en,testpc*en[:,None]]))}
 models['count_only']=(dn[:,None],en[:,None])
 models['PC3_count_additive']=(np.column_stack([trainpc,dn]),np.column_stack([testpc,en]))
 out={'pooled_mean':np.full(len(te),tr.y.mean()),'task_family_mean':np.array([tr.loc[tr.task_family==tf,'y'].mean() if (tr.task_family==tf).any() else tr.y.mean() for tf in te.task_family])}
 for m,(xx,xt) in models.items():out[m]=regress(xx,tr.y.to_numpy(),xt,penalty=1)[0]
 chance_train=1/tr.n.to_numpy();chance_test=1/te.n.to_numpy()
 relative=(tr.y.to_numpy()-chance_train)/(1-chance_train)
 cpred=regress(trainpc,relative,testpc,penalty=1)[0]
 out['PC3_chance_floor']=chance_test+(1-chance_test)*cpred
 for m,pp in out.items():
  for (_,r),p in zip(te.iterrows(),pp):predrows.append({'axis':axis,'fold':f,'model':r.Model,'model_family':r['Model Family'],'task_family':r.task_family,'objects':r.n,'predictor':m,'observed':r.y,'predicted':p})
pd.DataFrame(predrows).to_csv(O/'descriptor_predictions.csv',index=False)
df=pd.DataFrame(predrows);summary=[]
for (axis,m),g in df.groupby(['axis','predictor']):summary.append({'axis':axis,'predictor':m,'cells':len(g),**metrics(g.observed.to_numpy(),g.predicted.to_numpy())})
save('descriptor_summary.json',{'models':int(a.Model.nunique()),'model_families':int(a['Model Family'].nunique()),'task_families':2,'tasks':6,'observed_cells':len(a),'results':summary})
# Paired family resampling uncertainty for predictive loss differences; folds already fitted.
wide=df.pivot(index=['axis','fold','model','model_family','task_family','objects','observed'],columns='predictor',values='predicted').reset_index(); uncertainty=[]
for axis,g in wide.groupby('axis'):
 fams=g.model_family.unique()
 for control in ['PC1','PC3','count_only','PC3_count_additive','PC3_chance_floor']:
  losses=(g.PC3_count-g.observed)**2-(g[control]-g.observed)**2
  draws=[]
  for i in range(2000):
   ix=np.concatenate([np.flatnonzero(g.model_family.to_numpy()==f) for f in RNG.choice(fams,len(fams),replace=True)]);draws.append(float(losses.to_numpy()[ix].mean()))
  pd.DataFrame({'replicate':np.arange(2000),'delta_mse':draws}).to_csv(O/f'descriptor_fixed_bootstrap_{axis}_{control}.csv',index=False)
  uncertainty.append({'axis':axis,'comparator':control,'delta_mse':float(losses.mean()),'cluster_percentile95':np.quantile(draws,[.025,.975]).tolist(),'caveat':'resamples model families of fixed out-of-fold predictions; two target families do not establish target-population uncertainty'})
save('descriptor_uncertainty.json',uncertainty)
print(json.dumps({'association':summaries,'descriptor':summary},indent=2))
