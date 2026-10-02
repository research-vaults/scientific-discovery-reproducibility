"""Exact constructed examples, not fitted capability or biomedical forecasts."""
from pathlib import Path
import json, math
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P = Path(__file__).resolve().parents[1]
O = P/'evidence/capability_horizon_analysis'; O.mkdir(exist_ok=True)
F = P/'manuscript/figures'; F.mkdir(exist_ok=True)
q, eta0, v = .8, -3., 1.
h = float(logit(q)-eta0)
checks=[]
def check(name, condition):
    if not bool(condition): raise AssertionError(name)
    checks.append(name)
rows=[]
for lam in [1., .5, 0.]:
    analytic=h/(lam*v) if lam>0 else math.inf
    root=float(brentq(lambda t: expit(eta0+lam*v*t)-q,0,20)) if lam>0 else None
    if root is not None:
        check(f'analytic equals independent numerical root {lam}',abs(root-analytic)<1e-10)
        check(f'crossing bracket {lam}',expit(eta0+lam*v*(analytic-1e-6))<q<expit(eta0+lam*v*(analytic+1e-6)))
    else:check('zero coupling stays below threshold', expit(eta0)<q)
    rows.append(dict(coupling=lam,initial_probability=float(expit(eta0)),horizon=analytic if math.isfinite(analytic) else None,horizon_status='finite' if root is not None else 'infinite under this scenario',numerical_root=root))
check('common initial probability',len(set(x['initial_probability'] for x in rows))==1)
gated=max(6.,h)
check('gate determines crossing',gated==6. and expit(eta0+6)>q)
check('positive coupling interval',[h,h/.5]==[rows[0]['horizon'],rows[1]['horizon']])
for lam in np.linspace(.5,1,101):check(f'interval inclusion {lam:.3f}',h-1e-12<=h/lam<=h/.5+1e-12)
# Inverse-sensitivity equality examples: an intercept error shifts crossing by eps/m.
for m in [1.,.1]:
    eps=.2;true=h/m;estimated=(h-eps)/m
    check(f'time-error bound attained slope {m}',abs(abs(estimated-true)-eps/m)<1e-12)
# Common-observation mechanism counterexample; grid verifies illustration, not proof.
for r in np.linspace(0,1,101):check(f'identification bound {r:.2f}',min(r,1-r)<=.5)
# Formula with uncertain positive baseline gap, rate and coupling: check corners.
for hh in [4.,5.]:
 for lam in [.5,1.]:
  for rate in [.8,1.2]:check(f'box bound {hh,lam,rate}',4/1.2-1e-12<=hh/(lam*rate)<=5/(.5*.8)+1e-12)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.0,2.85),layout='constrained')
t=np.linspace(0,12,601)
colors=['#205693','#a45b16','#666666']
for lam,col in zip([1.,.5,0.],colors):
 axs[0].plot(t,expit(eta0+lam*t),color=col,label=rf'$\lambda={lam:g}$',lw=1.8)
axs[0].plot([0,6],[0,0],color='#19846d',ls=':',lw=2)
t2=np.linspace(6,12,301)
axs[0].plot(t2,expit(eta0+t2),color='#19846d',ls=':',lw=2,label='Access at year 6')
axs[0].plot([6,6],[0,expit(eta0+6)],color='#19846d',ls=':',lw=1)
axs[0].axhline(q,color='black',ls='--',lw=.8)
axs[0].set(xlim=(0,12),ylim=(-.035,1.03),xlabel='Hypothetical years from origin',ylabel='Target success probability',title='Same source trajectory')
axs[0].legend(loc='lower right',fontsize=8,frameon=True)
axs[0].text(.25,.83,'Required reliability 0.8',fontsize=8)
lams=np.linspace(.2,1.2,401)
axs[1].plot(lams,h/lams,color='#205693',lw=1.8)
axs[1].axvspan(.5,1,alpha=.15,color='#205693')
axs[1].scatter([.5,1],[h/.5,h],color='#205693',s=24)
axs[1].annotate('8.77 years',(.5,h/.5),xytext=(.58,11),fontsize=9,arrowprops={'arrowstyle':'-','lw':.7})
axs[1].annotate('4.39 years',(1,h),xytext=(.66,2.2),fontsize=9,arrowprops={'arrowstyle':'-','lw':.7})
axs[1].set(xlabel=r'Target coupling $\lambda$',ylabel='Capability horizon (scenario-years)',title='A bound needs a transfer relation',ylim=(0,23),xlim=(.2,1.2))
for ax in axs:ax.grid(alpha=.14)
fig.savefig(F/'capability_horizons.pdf');fig.savefig(O/'capability_horizons.png',dpi=180);plt.close(fig)
result={'status':'PASS','evidence_type':'constructed scenarios, no empirical fit','q':q,'eta0':eta0,'source_rate':v,'scenarios':rows,'access_gated_horizon':gated,'coupling_interval':[.5,1.],'conditional_horizon_interval':[h,h/.5],'time_error_examples':[{'log_odds_error':.2,'minimum_rate':m,'time_bound':.2/m} for m in [1.,.1]],'checks_passed':len(checks),'checks':checks}
(O/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:val for k,val in result.items() if k!='checks'},indent=2))
