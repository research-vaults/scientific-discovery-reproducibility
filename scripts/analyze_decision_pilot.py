"""Exact operating characteristics of a fixed, hypothetical two-arm pilot.

No biomedical data, stochastic simulation, fitted scaling law, or new estimator.
"""
from pathlib import Path
import json, math
import numpy as np
import scipy
from scipy.stats import beta, binom, binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
OUT=P/'evidence/decision_pilot';OUT.mkdir(exist_ok=True)
Q=.9; ALPHA=.05; NS=[25,50,100,250,500,1000]

def budget(p,cost):
    p=np.asarray(p,dtype=float)
    out=np.full(p.shape,np.inf)
    middle=(p>0)&(p<1)
    out[middle]=cost*np.ceil(np.log1p(-Q)/np.log1p(-p[middle]))
    out[p==1]=cost
    return out

def rules(n):
    x=np.arange(n+1)
    # Each arm interval has coverage >=1-alpha/2; each tail alpha/4.
    lo=np.zeros(n+1);hi=np.ones(n+1)
    lo[1:]=beta.ppf(ALPHA/4,x[1:],n-x[1:]+1)
    hi[:-1]=beta.ppf(1-ALPHA/4,x[:-1]+1,n-x[:-1])
    for k in sorted(set([0,1,n//2,n-1,n])):
        ci=binomtest(k,n).proportion_ci(1-ALPHA/2,method='exact')
        assert abs(lo[k]-ci.low)<1e-10 and abs(hi[k]-ci.high)<1e-10
    # Matrix rows baseline counts; columns new-procedure counts.
    adopt=budget(lo,11)[None,:] < budget(hi,10)[:,None]
    reject=budget(hi,11)[None,:] >= budget(lo,10)[:,None]
    assert not np.any(adopt & reject)
    plugin=budget(x/n,11)[None,:] < budget(x/n,10)[:,None]
    return lo,hi,adopt,reject,plugin

def summarize(n,p0,p1,r):
    lo,hi,adopt,reject,plugin=r
    w0=binom.pmf(np.arange(n+1),n,p0);w1=binom.pmf(np.arange(n+1),n,p1)
    assert abs(w0.sum()-1)<1e-10 and abs(w1.sum()-1)<1e-10
    # Each decision matrix is monotone in the candidate success count.
    # Sum its binomial tail per baseline count, avoiding dense matrix products.
    def tail_prob(mask,upper):
        count=mask.sum(axis=1)
        expected=(np.arange(n+1)[None,:] >= (n+1-count)[:,None]) if upper else (np.arange(n+1)[None,:] < count[:,None])
        assert np.array_equal(mask,expected)
        tails=binom.sf(n-count,n,p1) if upper else binom.cdf(count-1,n,p1)
        result=float(np.dot(w0,tails))
        if n==25:
            direct=float(np.sum(w0[:,None]*w1[None,:]*mask))
            assert abs(result-direct)<1e-12
        return result
    pa,pr,pp=tail_prob(adopt,True),tail_prob(reject,False),tail_prob(plugin,True)
    b0,b1=float(budget(p0,10)),float(budget(p1,11));saving=b1<b0
    wrong=pr if saving else pa
    assert wrong<=ALPHA+1e-10
    coverage0=float(w0[(lo<=p0)&(p0<=hi)].sum());coverage1=float(w1[(lo<=p1)&(p1<=hi)].sum())
    assert coverage0>=1-ALPHA/2-1e-10 and coverage1>=1-ALPHA/2-1e-10
    return dict(N_per_arm=n,p0=p0,p1=p1,true_baseline_budget=b0,true_candidate_budget=b1,true_saving=saving,
                interval_adopt=pa,interval_non_saving=pr,interval_unresolved=max(0.,1-pa-pr),
                interval_wrong_resolved=wrong,plugin_adopt=pp,plugin_wrong_decision=(1-pp if saving else pp),
                plugin_false_adoption=(0. if saving else pp),pilot_cost=21*n,
                campaigns_to_offset_pilot_in_reserved_budgets=(math.ceil(21*n/(b0-b1)) if saving else None))
rows=[];stress=[]
for n in NS:
    r=rules(n)
    for p1 in [.08,.10,.12,.20,.30]: rows.append(summarize(n,.1,p1,r))
    if n in [25,100,500]:
        for p0 in [.05,.1,.2]:
            for p1 in np.linspace(.02,.4,41):stress.append(summarize(n,p0,float(p1),r))
assert len(rows)==30 and len(stress)==369
# Direct probability checks use known parameters, independently of interval code.
for row in rows:
    for p,c,b in [(row['p0'],10,row['true_baseline_budget']),(row['p1'],11,row['true_candidate_budget'])]:
        n=int(b/c);assert 1-(1-p)**n>=Q-1e-12;assert 1-(1-p)**(n-1)<Q+1e-12
summary=dict(status='PASS',analysis_type='exact binomial enumeration under stipulated assumptions',q=Q,alpha=ALPHA,
             method='Two equal-tailed 97.5% Clopper–Pearson intervals; separate budget bounds; abstain unless separated',
             scipy_version=scipy.__version__,rows=rows,stress_grid_cases=len(stress),max_wrong_resolved_on_stress_grid=max(x['interval_wrong_resolved'] for x in stress),
             limitations=['independent constant-yield pilot and future attempts','fixed sample size, no optional stopping','known costs and unchanged validation','no setup cost or credit for pilot hits','not measured biology or proof of framework advantage'])
(OUT/'stress_grid.json').write_text(json.dumps(stress,indent=2)+'\n')
(OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
table=[r'\begin{tabular}{rrrrrrr}',r'\toprule',r'$N$ & $p_1$ & $B_1$ & Interval adopt (\%) & Unresolved (\%) & Point adopt (\%) & Pilot cost\\',r'\midrule']
for idx,row in enumerate(rows):
    if idx and idx%5==0:table.append(r'\midrule')
    table.append(f"{row['N_per_arm']} & {row['p1']:.2f} & {row['true_candidate_budget']:.0f} & {100*row['interval_adopt']:.3f} & {100*row['interval_unresolved']:.3f} & {100*row['plugin_adopt']:.3f} & {row['pilot_cost']}"+r'\\')
table.extend([r'\bottomrule',r'\end{tabular}'])
(P/'manuscript/figures/decision_pilot_table.tex').write_text('\n'.join(table)+'\n')
main_table=[r'\begin{tabular}{rrrrr}',r'\toprule',r'$N$ per arm & Candidate $p_1$ & Interval adopt & Unresolved & Point adopt\\',r'\midrule']
for n,p1 in [(100,.10),(100,.20),(1000,.12),(1000,.20)]:
    row=next(x for x in rows if x['N_per_arm']==n and x['p1']==p1)
    main_table.append(f"{n} & {p1:.2f} & {100*row['interval_adopt']:.3f} & {100*row['interval_unresolved']:.3f} & {100*row['plugin_adopt']:.3f}"+r'\\')
main_table.extend([r'\bottomrule',r'\end{tabular}'])
(P/'manuscript/figures/decision_pilot_main_table.tex').write_text('\n'.join(main_table)+'\n')

plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.1,2.65),layout='constrained')
for p1,color,label in [(.12,'#b45309','Small gain: p₁ = 0.12'),(.2,'#0072B2','Larger gain: p₁ = 0.20')]:
    sub=[x for x in rows if x['p1']==p1]
    axs[0].plot(NS,[100*x['interval_adopt'] for x in sub],'-o',color=color,label=label.replace('p₁',r'$p_1$'),markersize=3)
axs[0].set(xscale='log',xlabel='Pilot attempts per procedure',ylabel='Interval-rule adoption (%)',ylim=(-2,102),title='(a) Resolving a true saving')
axs[0].set_xticks([25,100,500,1000],labels=['25','100','500','1000']);axs[0].legend(fontsize=8,loc='upper left')
sub=[x for x in rows if x['p1']==.1]
axs[1].plot(NS,[100*x['plugin_false_adoption'] for x in sub],'-o',color='#D55E00',label='Point-estimate rule',markersize=3)
axs[1].plot(NS,[100*x['interval_adopt'] for x in sub],'-s',color='#0072B2',label='Interval rule',markersize=3)
axs[1].axhline(5,color='#777777',linestyle=':',linewidth=1,label='5% error bound (interval rule)')
axs[1].set(xscale='log',xlabel='Pilot attempts per procedure',ylabel='False adoption probability (%)',ylim=(-1,70),title='(b) Unchanged yield, higher cost')
axs[1].set_xticks([25,100,500,1000],labels=['25','100','500','1000']);axs[1].legend(fontsize=8,loc='upper right')
for ax in axs:ax.grid(alpha=.15)
fig.savefig(P/'manuscript/figures/decision_pilot.pdf');fig.savefig(OUT/'decision_pilot.png',dpi=180);plt.close(fig)
print(json.dumps({'status':'PASS','selected':[x for x in rows if x['N_per_arm'] in [100,500,1000] and x['p1'] in [.1,.12,.2]],'stress_max':max(x['interval_wrong_resolved'] for x in stress)},indent=2))
