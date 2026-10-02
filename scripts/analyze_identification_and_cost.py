"""Exact finite-sample and finite-domain calculations; no model or network calls."""
from pathlib import Path
import json, itertools, math
import numpy as np
from scipy.stats import beta,binom
from scipy.special import logit,expit
P=Path(__file__).resolve().parents[1];O=P/'evidence/review_remediation_2026-09-19/results';O.mkdir(exist_ok=True)
def save(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def cp(n,alpha=.05):
 x=np.arange(n+1);l=beta.ppf(alpha/4,x,n-x+1);u=beta.ppf(1-alpha/4,x+1,n-x);l[0]=0;u[-1]=1
 return l,u
rec=[]
for n in [25,100,500]:
 l,u=cp(n);lower=logit(l)[None,:]-logit(u)[:,None];upper=logit(u)[None,:]-logit(l)[:,None]
 for p0,delta,lam in itertools.product([.01,.05,.2],[.5,1,2],[0,.5,1]):
  p1=expit(logit(p0)+lam*delta); w=np.outer(binom.pmf(np.arange(n+1),n,p0),binom.pmf(np.arange(n+1),n,p1));lo=lower/delta;hi=upper/delta
  cover=float(w[(lo<=lam+1e-12)&(hi>=lam-1e-12)].sum());power=float(w[lo>0].sum());width=hi-lo;idx=np.argsort(width.ravel());cum=np.cumsum(w.ravel()[idx]);med=width.ravel()[idx[np.searchsorted(cum,.5)]]
  assert cover>=.95-1e-10
  point=(logit((np.arange(n+1)+.5)/(n+1))[None,:]-logit((np.arange(n+1)+.5)/(n+1))[:,None])/delta
  bias=float(np.sum(w*(point-lam)));rmse=float(np.sqrt(np.sum(w*(point-lam)**2)))
  rec.append({'point_estimator_bias':bias,'point_estimator_rmse':rmse,'N_per_level':n,'p0':p0,'Delta':delta,'lambda':lam,'p1':float(p1),'coverage':cover,'positive_lower_probability':power,'infinite_probability':float(w[~np.isfinite(width)].sum()),'median_width':float(med) if np.isfinite(med) else 'infinity','cost_at_unit_price':2*n})
save('identification.json',rec)
# Independent direct beta interval calls and joint sums on smallest design.
maxdiff=0
for r in [r for r in rec if r['N_per_level']==25]:
 n=25;coverage=0
 for i,j in itertools.product(range(n+1),repeat=2):
  l0=0 if i==0 else beta.ppf(.0125,i,n-i+1);u0=1 if i==n else beta.ppf(.9875,i+1,n-i)
  l1=0 if j==0 else beta.ppf(.0125,j,n-j+1);u1=1 if j==n else beta.ppf(.9875,j+1,n-j)
  lo=(logit(l1)-logit(u0))/r['Delta'];hi=(logit(u1)-logit(l0))/r['Delta']
  if lo-1e-12<=r['lambda']<=hi+1e-12:coverage+=binom.pmf(i,n,r['p0'])*binom.pmf(j,n,r['p1'])
 maxdiff=max(maxdiff,abs(coverage-r['coverage']))
assert maxdiff<1e-12
# Exact program verification on every input, no Monte Carlo.
synthesis=[]
for n in range(2,9):
 m=2**n;candidates=list(range(1,m,2));a0=candidates[len(candidates)//2];x=np.arange(m);target=a0*x%m
 matches=[a for a in candidates if np.array_equal(a*x%m,target)];assert matches==[a0];N=len(candidates);cap=math.ceil(.9*N)
 assert cap/N>=.9 and (cap-1)/N<.9
 synthesis.append({'bits':n,'programs':N,'input_domain':m,'hidden_coefficient':a0,'full_verifier_valid_count':len(matches),'verification_evaluations_per_call':m,'cap_q90':cap,'reliable_work':cap*m,'mean_random_order_calls':(N+1)/2,'informative_observations':1,'informative_total_evaluations':1+m,'verified_candidate':a0})
save('synthesis.json',synthesis)
# The code counts complete verification (no early exit) and one output-producing observation.
# Pilot decision arrays; cross-arm dependence leaves marginal CP coverage intact.
def budget(p,a):
 p=np.asarray(p);out=np.full(p.shape,np.inf);mid=(p>0)&(p<1);out[mid]=a*np.ceil(np.log(.1)/np.log1p(-p[mid]));out[p==1]=a
 return out

def decisions(n,alpha):
 l,u=cp(n,alpha);ad=budget(l,11)[None,:]<budget(u,10)[:,None];no=budget(u,11)[None,:]>=budget(l,10)[:,None]
 assert not np.any(ad&no)
 return ad,no

def evolve(p0,p1,r,looks,alpha):
 state=np.ones((1,1));adopt=reject=cost=0.;res=[]
 for n in range(1,max(looks)+1):
  cost+=21*state.sum();nxt=np.zeros((n+1,n+1));nxt[:-1,:-1]+=state*(1-p0-p1+r);nxt[1:,:-1]+=state*(p0-r);nxt[:-1,1:]+=state*(p1-r);nxt[1:,1:]+=state*r;state=nxt
  if n in looks:
   a,b=decisions(n,alpha);pa=float(state[a].sum());pr=float(state[b].sum());adopt+=pa;reject+=pr;state[a|b]=0;res.append({'N':n,'stop_adopt':pa,'stop_non_saving':pr,'remaining':float(state.sum())})
 assert abs(adopt+reject+state.sum()-1)<1e-11
 wrong=adopt if budget(np.array(p1),11)>=budget(np.array(p0),10) else reject
 assert wrong<=.05+1e-10
 return {'adopt':adopt,'non_saving':reject,'unresolved':float(state.sum()),'expected_pilot_cost':float(cost),'wrong':wrong,'looks':res}
pilots=[]
for p1 in [.08,.1,.12,.2,.3]:
 for dep,r in [('lower',max(0,.1+p1-1)),('independent',.1*p1),('upper',min(.1,p1))]:
  for design,looks,alpha in [('fixed',[100],.05),('sequential',[25,50,100],.05/3)]:
   pilots.append({'p0':.1,'p1':p1,'pair_dependence':dep,'joint_success':r,'design':design,**evolve(.1,p1,r,looks,alpha)})
save('pilot_extensions.json',pilots)
voi=[];ps=np.array([.1,.2,.3]);prior=np.ones(3)/3;loss=budget(ps,11);base=220
for n in [0,25,50,100]:
 probs=binom.pmf(np.arange(n+1)[:,None],n,ps[None,:]);joint=probs*prior;px=joint.sum(1);post=joint/px[:,None];expected=post@loss
 risk=float(np.dot(px,np.minimum(base,expected)));no=min(base,float(prior@loss));perfect=float(prior@np.minimum(base,loss))
 for k in [1,10,25,100]:voi.append({'N_candidate_only':n,'K_campaigns':k,'no_pilot_expected_reserve':k*no,'posterior_optimal_expected_reserve':k*risk,'pilot_cost':11*n,'net_value':k*(no-risk)-11*n,'perfect_information_upper_value':k*(no-perfect)})
save('value_of_information.json',voi)
worked=[]
for i,j in [(10,40),(10,10),(30,5)]:
 l,u=cp(100);a,b=decisions(100,.05)
 worked.append({'x0':i,'x1':j,'N':100,'interval0':[float(l[i]),float(u[i])],'interval1':[float(l[j]),float(u[j])],'budget0':[float(budget(u[i],10)),float(budget(l[i],10))],'budget1':[float(budget(u[j],11)),float(budget(l[j],11))],'decision':'adopt' if a[i,j] else 'non-saving' if b[i,j] else 'unresolved'})
save('worked_pilots.json',worked)
sens=[]
for q,lam,v,gate in itertools.product([.5,.8,.95],[0,.25,.5,1,2],[.25,1,2],[0,2,6]):
 t=max(gate,(logit(q)+3)/(lam*v)) if lam else math.inf
 sens.append({'q':q,'lambda':lam,'v':v,'access':gate,'horizon':t if math.isfinite(t) else 'infinity'})
save('horizon_sensitivity.json',sens)
# Cost sensitivity across all former hand-picked economic parameters.
costsens=[]
for q,c,v,p in itertools.product([.5,.9,.99],[1,2,4],[0,1,9,99],[.08,.1,.12,.2,.3]):
 base=(1+v)*math.ceil(math.log1p(-q)/math.log1p(-.1));new=(c+v)*math.ceil(math.log1p(-q)/math.log1p(-p));costsens.append({'q':q,'proposal_cost':c,'verification_cost':v,'yield':p,'baseline_reserve':base,'candidate_reserve':new,'saves':new<base})
save('cost_sensitivity.json',costsens)
save('analytic_checks.json',{'identification_settings':len(rec),'independent_direct_max_difference':maxdiff,'synthesis_sizes':len(synthesis),'pilot_settings':len(pilots),'horizon_settings':len(sens),'cost_settings':len(costsens),'time_error_sensitivity':[{'epsilon':e,'m':m,'time_error_bound':e/m} for e,m in itertools.product([.05,.2,.5],[.05,.1,1])],'completion_reversal':{'old_capability_horizon':1,'new_capability_horizon':0,'old_mean_completion':1+1/.8,'new_mean_completion':10/.9}})
print('Exact analyses complete',len(rec),'identification settings;',len(pilots),'pilot settings')
