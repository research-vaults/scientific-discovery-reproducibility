"""Post-review exact threshold sensitivity; no new empirical evidence.
Run with --out PATH. Contract lives beside outputs, frozen before execution.
Only standard-library arithmetic; independent of the previous SciPy implementation.
"""
import argparse, csv, hashlib, json, math
from pathlib import Path

N = 100

def pmf(n, p):
    if p == 0: return [1.] + [0.] * n
    if p == 1: return [0.] * n + [1.]
    logs = [math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)+k*math.log(p)+(n-k)*math.log1p(-p) for k in range(n+1)]
    values = [math.exp(x) for x in logs]
    assert abs(math.fsum(values)-1) < 1e-11
    return values

def lower(k, n, tail):
    if not k: return 0.
    a,b=0.,1.
    for _ in range(55):
        m=(a+b)/2
        if math.fsum(pmf(n,m)[k:]) < tail: a=m
        else: b=m
    answer=(a+b)/2
    assert abs(math.fsum(pmf(n,answer)[k:])-tail)<1e-11
    return answer

def intervals(n, alpha):
    lo=[lower(k,n,alpha/4) for k in range(n+1)]
    hi=[1-lo[n-k] for k in range(n+1)]
    assert abs(hi[0]-(1-(alpha/4)**(1/n)))<1e-12
    return lo,hi

def reserve(p,c,q):
    if p<=0: return math.inf
    if p>=1: return c
    return c*math.ceil(math.log1p(-q)/math.log1p(-p))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'evidence/frontieraudit_transfer_2026-09-24_014611');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    weights={p:pmf(N,p) for p in [.1,.12,.2,.3]}
    rows=[]
    for alpha in [.01,.05,.10]:
        lo,hi=intervals(N,alpha)
        for p in weights:
            coverage=math.fsum(w for k,w in enumerate(weights[p]) if lo[k]<=p<=hi[k]);assert coverage>=1-alpha/2-1e-10
        for q in [.8,.9,.95]:
            b0l=[reserve(x,10,q) for x in hi];b0u=[reserve(x,10,q) for x in lo]
            b1l=[reserve(x,11,q) for x in hi];b1u=[reserve(x,11,q) for x in lo]
            for p1 in [.1,.12,.2,.3]:
                adopt=[];reject=[];plugin=[]
                for i,w0 in enumerate(weights[.1]):
                    for j,w1 in enumerate(weights[p1]):
                        a=b1u[j]<b0l[i];r=b1l[j]>=b0u[i];assert not(a and r)
                        w=w0*w1
                        if a:adopt.append(w)
                        if r:reject.append(w)
                        if reserve(j/N,11,q)<reserve(i/N,10,q):plugin.append(w)
                a=math.fsum(adopt);r=math.fsum(reject);b0=reserve(.1,10,q);b1=reserve(p1,11,q);saving=b1<b0
                wrong=r if saving else a
                assert wrong<=alpha+1e-10
                rows.append(dict(N=N,q=q,alpha=alpha,p0=.1,p1=p1,adopt=a,non_saving=r,defer=1-a-r,wrong=wrong,error_among_resolved=wrong/(a+r) if a+r else None,plugin_adopt=math.fsum(plugin),baseline_reserve=b0,candidate_reserve=b1,reserve_saving=b0-b1,pilot_cost=2100,optimistic_campaigns_to_offset_pilot=math.ceil(2100/(b0-b1)) if saving else None))
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'evidence/decision_pilot/results.json').read_text())['rows']
    residuals=[]
    for row in rows:
        if row['q']==.9 and row['alpha']==.05:
            ref=next(x for x in old if x['N_per_arm']==100 and x['p1']==row['p1'])
            for key,other in [('adopt','interval_adopt'),('defer','interval_unresolved'),('plugin_adopt','plugin_adopt')]:
                residuals.append(abs(row[key]-ref[other]));assert residuals[-1]<1e-10
    assert len(rows)==36
    (args.out/'results.json').write_text(json.dumps({'status':'PASS','type':'exact hypothetical sensitivity, not empirical validation','rows':rows,'old_result_max_absolute_difference':max(residuals)},indent=2)+'\n')
    with (args.out/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    manifest={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(__file__),args.out/'CONTRACT.md',args.out/'results.json',args.out/'results.csv']}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for row in rows:
        if row['p1'] in [.1,.2]:print('q',row['q'],'alpha',row['alpha'],'p1',row['p1'],'adopt',round(100*row['adopt'],4),'defer',round(100*row['defer'],4))
    print('PASS:36 cells; independent binomial inversion; saved-results agreement',max(residuals))
if __name__=='__main__':main()
