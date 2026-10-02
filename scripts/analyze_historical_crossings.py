"""Retrospective diagnostic on a single METR task-suite version."""
from pathlib import Path
import yaml,json,datetime
import numpy as np
P=Path(__file__).resolve().parents[1];E=P/'evidence/review_remediation_2026-09-19';O=E/'results';O.mkdir(exist_ok=True)
s=yaml.safe_load((E/'sources/metr_results.yaml').read_text());origin=datetime.date(2019,1,1);cut=datetime.date(2024,6,1)
rows=[]
for name,r in s['results'].items():
 d=r['release_date'];d=datetime.date.fromisoformat(d) if isinstance(d,str) else d
 h=r['metrics']['p50_horizon_length']['estimate']
 rows.append({'model':name,'date':d.isoformat(),'t':(d-origin).days/365.25,'minutes':h,'train':d<cut})
rows.sort(key=lambda x:(x['date'],x['model']))
x=np.array([r['t'] for r in rows]);y=np.log([r['minutes'] for r in rows]);train=np.array([r['train'] for r in rows]);coef=np.polyfit(x[train],y[train],1)
eps=float(np.max(np.abs(y[train]-np.polyval(coef,x[train]))))
# Training heuristic uses minimum strictly positive increment; negative increments are disclosed.
dates=np.unique(x);z=np.array([y[x==t].max() for t in dates]);slopes=np.diff(z)/np.diff(dates);early=dates[1:]<(cut-origin).days/365.25
pos=slopes[early&(slopes>0)];m=float(pos.min());heur=eps/m
# Running maximum is a monotone selection transformation, not latent continuous ground truth.
front=np.maximum.accumulate(z);frontslopes=np.diff(front)/np.diff(dates)
results=[]
for q in [15,30,60,120]:
 threshold=np.log(q);pred=float((threshold-coef[1])/coef[0]);seen=[r for r in rows if not r['train'] and r['minutes']>=q]
 r={'threshold_minutes':q,'predicted_date':(origin+datetime.timedelta(days=pred*365.25)).isoformat(),'first_record_date':seen[0]['date'] if seen else None,'censored':not bool(seen),'heuristic_years':heur}
 if seen:r.update({'error_to_first_record_years':abs(seen[0]['t']-pred),'heuristic_covers_first_record':abs(seen[0]['t']-pred)<=heur})
 idx=np.flatnonzero(front>=threshold)
 if len(idx) and idx[0]>0:
  k=idx[0];actual=dates[k-1]+(threshold-front[k-1])/frontslopes[k-1];left=min(actual,pred);right=max(actual,pred)
  r['interpolated_frontier_crossing_date']=(origin+datetime.timedelta(days=float(actual)*365.25)).isoformat()
  if left<dates[0] or right>dates[-1]:r['oracle_status']='crossing outside observed interpolation domain'
  else:
   use=(dates[:-1]<right)&(dates[1:]>left);ms=frontslopes[use];r['oracle_min_slope']=float(ms.min()) if len(ms) else None
   if len(ms) and ms.min()>0:
    knots=np.r_[left,dates[(dates>left)&(dates<right)],right];true=np.interp(knots,dates,front);oracleeps=float(np.max(np.abs(true-np.polyval(coef,knots))));bound=oracleeps/ms.min();error=abs(actual-pred)
    assert error<=bound+1e-10
    r.update({'oracle_status':'conditional algebra verified on selected interpolation','oracle_epsilon':oracleeps,'oracle_bound_years':float(bound),'interpolated_error_years':float(error)})
   else:r['oracle_status']='positive slope premise fails (frontier plateau)'
 else:r['oracle_status']='unbracketed'
 results.append(r)
out={'benchmark':s['benchmark_name'],'version':s['long_tasks_version'],'records':len(rows),'train_records':int(train.sum()),'test_records':int((~train).sum()),'cutoff':str(cut),'log_minutes_slope_per_year':float(coef[0]),'training_max_residual':eps,'training_min_positive_increment':m,'training_nonpositive_increments':int(np.sum(slopes[early]<=0)),'all_nonpositive_increments':int(np.sum(slopes<=0)),'diagnostics':results,'all_records':rows,'scope':'Retrospective current-release data indexed by model release date; measurement availability at cutoff not established. No prospective forecast, latent-curve bound or empirical proof of the lemma.'}
(O/'historical_crossings.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='all_records'},indent=2))
