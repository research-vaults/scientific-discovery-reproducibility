"""Matched-guarantee directed certificates and implementable reserve decisions."""
from pathlib import Path
import ast,json,math
from functools import lru_cache
import numpy as np
from scipy.stats import binom,beta,binomtest
P=Path(__file__).resolve().parents[1];E=P/'evidence/eleven_review_remediation_2026-09-21_001619';O=E/'results';O.mkdir(exist_ok=True)
Q=.9;ALPHA=.05
tr=ast.parse((P/'scripts/analyze_decision_pilot.py').read_text())
exec(compile(ast.Module(body=[n for n in tr.body if isinstance(n,ast.FunctionDef) and n.name in {'budget','rules','summarize'}],type_ignores=[]),'existing_pilot_helpers','exec'))

def theta(n):
 return 1. if n<=0 else -np.expm1(np.log1p(-Q)/n)

@lru_cache(None)
def threshold_test(n,p,high,alpha):
 """Return rejection-count boundary for an exact one-sided binomial test."""
 x=np.arange(n+1);pv=binom.sf(x-1,n,p) if high else binom.cdf(x,n,p)
 ok=x[pv<=alpha+1e-15]
 return (int(ok.min()) if len(ok) else n+1) if high else (int(ok.max()) if len(ok) else -1)

def direction_table(N,alpha=.05):
 d=N//5;v=N-d;out=[]
 # Smoothed DESIGN estimates choose one direction and monetary separator only.
 for x0 in range(d+1):
  for x1 in range(d+1):
   b0=float(budget((x0+.5)/(d+1),10));b1=float(budget((x1+.5)/(d+1),11))
   if b1==b0:out.append((x0,x1,0,0,0,0.,0.));continue
   B=(b0+b1)/2
   if b1<b0:
    # B1 < B <= B0: candidate n <= ceil(B/11)-1; baseline n >= ceil(B/10).
    k1=math.ceil(B/11)-1;k0=math.ceil(B/10)-1
    hi1=threshold_test(v,theta(k1),True,alpha);lo0=threshold_test(v,theta(k0),False,alpha)
    out.append((x0,x1,1,lo0,hi1,theta(k0),theta(k1)))
   else:
    # B0 <= B <= B1: baseline n <= floor(B/10); candidate n >= ceil(B/11).
    k0=math.floor(B/10);k1=math.ceil(B/11)-1
    hi0=threshold_test(v,theta(k0),True,alpha);lo1=threshold_test(v,theta(k1),False,alpha)
    out.append((x0,x1,-1,hi0,lo1,theta(k0),theta(k1)))
 return d,v,np.array(out)

def directed(N,p0,p1,tab):
 d,v,t=tab;w=binom.pmf(t[:,0],d,p0)*binom.pmf(t[:,1],d,p1)
 aa=t[:,2]==1;nn=t[:,2]==-1
 pa=float(np.sum(w[aa]*binom.cdf(t[aa,3],v,p0)*binom.sf(t[aa,4]-1,v,p1)))
 pn=float(np.sum(w[nn]*binom.sf(t[nn,3]-1,v,p0)*binom.cdf(t[nn,4],v,p1)))
 b0,b1=float(budget(p0,10)),float(budget(p1,11));saving=b1<b0;wrong=pn if saving else pa
 assert wrong<=.05+1e-10
 # Deployment defaults to baseline after deferral; loss is true-yield reserve excess.
 regret=(1-pa)*max(0,b0-b1)+pa*max(0,b1-b0)
 return dict(N_per_arm=N,p0=p0,p1=p1,adopt=pa,non_saving=pn,unresolved=1-pa-pn,wrong=wrong,error_among_resolved=wrong/(pa+pn) if pa+pn else None,pilot_cost=21*N,selection_regret=regret)

@lru_cache(None)
def two_look_arm(v,theta0,p,high):
 e=v//2;r=v-e;x=np.arange(e+1)
 b1=threshold_test(e,theta0,high,.025);b2=threshold_test(v,theta0,high,.025)
 mask=x>=b1 if high else x<=b1
 early=binom.sf(b1-1,e,p) if high else binom.cdf(b1,e,p)
 final=binom.sf(b2-1,v,p) if high else binom.cdf(b2,v,p)
 remain=binom.sf(b2-x-1,r,p) if high else binom.cdf(b2-x,r,p)
 both=float(np.sum(binom.pmf(x,e,p)*mask*remain))
 return float(early),float(final),both

def sequential(N,p0,p1,tab):
 d,v,t=tab;pa=pn=stopped=0.
 for x0,x1,di,_,_,th0,th1 in t:
  if not di:continue
  w=binom.pmf(x0,d,p0)*binom.pmf(x1,d,p1)
  a=two_look_arm(v,float(th0),p0,di==-1);b=two_look_arm(v,float(th1),p1,di==1)
  pe=a[0]*b[0];pr=pe+a[1]*b[1]-a[2]*b[2]
  stopped+=w*pe
  if di==1:pa+=w*pr
  else:pn+=w*pr
 saving=float(budget(p1,11))<float(budget(p0,10));wrong=pn if saving else pa
 assert wrong<=.05+1e-10
 return dict(N_per_arm=N,p0=p0,p1=p1,adopt=float(pa),non_saving=float(pn),unresolved=float(1-pa-pn),wrong=float(wrong),pilot_cost=float(21*(N-(v-v//2)*stopped)))

rows=[];stress=[]
for n in [100,1000]:
 tab=direction_table(n);r=rules(n)
 for p1 in [.08,.1,.12,.2,.3]:
  old=summarize(n,.1,p1,r);d=directed(n,.1,p1,tab)
  old['error_among_resolved']=old['interval_wrong_resolved']/(1-old['interval_unresolved'])
  old['selection_regret']=(1-old['interval_adopt'])*max(0,old['true_baseline_budget']-old['true_candidate_budget'])+old['interval_adopt']*max(0,old['true_candidate_budget']-old['true_baseline_budget'])
  rows.append(dict(interval=old,directed=d,sequential=sequential(n,.1,p1,tab)))
 for p0 in [.05,.1,.2]:
  for p1 in np.linspace(.02,.4,41):stress.append(directed(n,p0,float(p1),tab))
 print('directed pilot',n,flush=True)
# Exact small-N brute enumeration independently checks factorized conditional probabilities.
tab=direction_table(10);d,v,t=tab;p0=.1;p1=.2;tot=0
for x0,x1,di,b0,b1,_,_ in t:
 if di!=1:continue
 for y0 in range(v+1):
  for y1 in range(v+1):
   if y0<=b0 and y1>=b1:tot+=binom.pmf(x0,d,p0)*binom.pmf(x1,d,p1)*binom.pmf(y0,v,p0)*binom.pmf(y1,v,p1)
assert abs(tot-directed(10,p0,p1,tab)['adopt'])<1e-12
# Posterior-predictive policy: actually choose a method and integer cap.
ps=np.array([.1,.2,.3]);caps=np.arange(1,1001);basecap=22;basebudget=220
vo=[];states=[]
for prior_name,prior in [('equal',np.array([1/3]*3)),('pessimistic',np.array([.8,.1,.1])),('optimistic',np.array([.1,.2,.7]))]:
 for n in [0,5,25,100]:
  like=binom.pmf(np.arange(n+1)[:,None],n,ps[None,:]);mass=like@prior
  avg_res=avg_spend=avg_future_success=0.;state_succ=np.zeros(3);state_spend=np.zeros(3);candidate_chosen=0.
  for k in range(n+1):
   post=prior*like[k]/mass[k]
   fail=(1-ps[None,:])**caps[:,None]@post
   cap=int(caps[np.flatnonzero(fail<=.1)[0]])
   cand=11*cap<basebudget;chosen=11*cap if cand else basebudget
   succ=1-(1-ps)**cap if cand else np.full(3,1-.9**22)
   spend=11*(1-(1-ps)**cap)/ps if cand else np.full(3,10*(1-.9**22)/.1)
   assert float(post@succ)>=.9-1e-12
   avg_res+=mass[k]*chosen;avg_spend+=mass[k]*(post@spend);avg_future_success+=mass[k]*(post@succ);candidate_chosen+=mass[k]*cand
   state_succ+=like[k]*succ;state_spend+=like[k]*spend
   if prior_name=='equal':states.append(dict(N=n,successes=k,posterior=post.tolist(),method='candidate' if cand else 'baseline',cap=cap if cand else 22,reserve=chosen,state_success=succ.tolist(),posterior_success=float(post@succ)))
  pilot_hit=float(prior@(1-(1-ps)**n))
  # One-hit objective: stop the pilot after a hit; if all fail use its posterior policy.
  # In failure branch k=0, all observations were taken; same posterior policy is valid.
  post0=prior*(1-ps)**n;post0/=post0.sum();f=(1-ps[None,:])**caps[:,None]@post0;cap0=int(caps[np.flatnonzero(f<=.1)[0]])
  ca0=11*cap0<basebudget;su0=1-(1-ps)**cap0 if ca0 else np.full(3,1-.9**22);sp0=11*su0/ps if ca0 else np.full(3,10*(1-.9**22)/.1)
  first_spend=float(prior@(11*(1-(1-ps)**n)/ps+(1-ps)**n*sp0));first_success=float(prior@(1-(1-ps)**n+(1-ps)**n*su0))
  vo.append(dict(prior=prior_name,N=n,expected_implementable_reserve=avg_res,expected_future_expenditure=avg_spend,prior_predictive_future_success=avg_future_success,state_future_success=state_succ.tolist(),fixed_pilot_cost=11*n,pilot_at_least_one_hit=pilot_hit,expected_first_hit_expenditure=first_spend,first_hit_success=first_success,candidate_choice_probability=candidate_chosen))
# Known no-information operational cap: 14 not mean of true-yield caps.
s0=next(x for x in states if x['N']==0);assert s0['cap']==14 and s0['reserve']==154
for name,obj in [('decision_robustness.json',rows),('decision_robustness_stress.json',stress),('operational_policy.json',vo),('operational_policy_actions.json',states)]:
 (O/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
print('decision robustness complete',flush=True)
