"""Stipulated resource profiles and scenario crossings, never empirical forecasts."""
from pathlib import Path
import json
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parents[1]
E = P / 'evidence/complexity_scenarios'
E.mkdir(exist_ok=True)
# All requirements belong to one stipulated procedure per task, at fixed quality.
# Compute and informative-data units are separately normalized to availability at the arbitrary origin.
profiles = {'A': (4, 1, .5), 'B': (2, 4, .5), 'C': (2, 2, 2)}
colors = {'A': '#176b91', 'B': '#b65b16', 'C': '#7e4587'}
def gap(key, year, dc=2., de=4., dl=math.inf):
    c, e, latency = profiles[key]
    dt = np.asarray(year)
    return np.maximum.reduce([c / 2**(dt/dc), e / 2**(dt/de),
                              latency / 2**(dt/dl)])
def crossing(key, dc=2., de=4., dl=math.inf):
    c, e, latency = profiles[key]
    feedback = 0 if latency <= 1 else (math.inf if math.isinf(dl) else dl*math.log2(latency))
    return max(0, dc*math.log2(c), de*math.log2(e), feedback)

# Analytic threshold horizons must agree with direct inequalities on both sides.
expected = [('A', 2., 4., math.inf, 4), ('B', 2., 4., math.inf, 8),
            ('C', 2., 4., math.inf, math.inf), ('C', 2., 4., 6., 6),
            ('B', 1., 4., math.inf, 8), ('B', 2., 2., math.inf, 4)]
for key, dc, de, dl, target in expected:
    assert crossing(key, dc, de, dl) == target
    if math.isfinite(target):
        assert abs(float(gap(key, target, dc, de, dl))-1) < 1e-12
        assert gap(key, target-.001, dc, de, dl) > 1
        assert gap(key, target+.001, dc, de, dl) <= 1
    else:
        assert profiles[key][2] > 1 and math.isinf(dl)
assert 17609 < 1000*math.log2(200000) < 17610

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig, axs = plt.subplots(1, 2, figsize=(8.5, 4.3), layout='constrained')
years = np.linspace(0, 10, 601)
ax = axs[0]
for key in profiles:
    ax.plot(years, gap(key, years), color=colors[key], lw=2,
            label=f'{key}: '+{'A':'compute demand','B':'data demand','C':'feedback delay'}[key])
ax.plot(years, gap('C', years, dl=6), color=colors['C'], ls='--', lw=2,
        label='C: feedback also improves')
for key, date, dl in [('A',4,math.inf),('B',8,math.inf),('C',6,6.)]:
    ax.scatter(date,1,s=30,color=colors[key],zorder=5)
    ax.annotate(f'{key}: {date}',(date,1),xytext=(0,-37),textcoords='offset points',
                ha='center',color=colors[key],fontsize=10)
ax.axhline(1,color='#333333',lw=1,ls=':')
ax.axhspan(.2,1,color='#428763',alpha=.08)
ax.set(yscale='log',ylim=(.2,5),xlim=(0,10),
       xlabel='Elapsed scenario years',
       ylabel='Largest requirement / available allowance',
       title='(a) Every constraint must pass')
ax.set_xticks([0,2,4,6,8,10])
ax.set_yticks([.25,.5,1,2,4],labels=['0.25','0.5','1','2','4'])
ax.legend(frameon=False,fontsize=10,loc='upper right')
ax.text(.2,.24,'At or below 1: feasible for the stipulated procedure',fontsize=9)

ax = axs[1]
variants=[('Reference',2,4),('Faster compute',1,4),
          ('Faster data',2,2)]
for i,(label,dc,de) in enumerate(variants):
    a,b = crossing('A',dc,de),crossing('B',dc,de)
    ax.plot([a,b],[i-.10,i+.10],color='#b8b8b8',lw=1,zorder=1)
    for key,date,offset in [('A',a,-.10),('B',b,.10)]:
        ax.scatter(date,i+offset,color=colors[key],s=45,zorder=3)
        ax.annotate(f'{key}: {int(date)}',(date,i+offset),xytext=(0,9 if key=='A' else -15),
                    textcoords='offset points',ha='center',fontsize=10,color=colors[key])
ax.set_yticks(range(3),labels=[v[0] for v in variants])
ax.set(xlim=(1,10),ylim=(-.55,2.55),xlabel='Years until first feasible start',
       title='(b) Which resource improves?')
ax.invert_yaxis()
ax.set_xticks([2,4,6,8,10])
ax.grid(axis='x',alpha=.15)
fig.suptitle('Hypothetical resource scenarios — invented tasks, no predicted breakthroughs',fontsize=11)
fig.savefig(P/'manuscript/figures/complexity_scenarios.pdf')
fig.savefig(E/'complexity_scenarios.png',dpi=190)
plt.close(fig)
(E/'scenario_checks.json').write_text(json.dumps({
    'kind':'exact stipulated scenario, not fitted or calibrated',
    'profiles_compute_data_feedback':profiles,
    'arbitrary_time_origin':0,'compute_doubling_years':2,'informative_data_doubling_years':4,
    'fixed_completion_deadline_years':1,'alternative_feedback_halving_years':6,
    'crossings':[{ 'task':key,'compute_doubling':dc,'data_doubling':de,
                  'feedback_halving':None if math.isinf(dl) else dl,
                  'years_until_first_feasible_start':None if math.isinf(t) else t}
                 for key,dc,de,dl,t in expected],
    'all_analytic_and_boundary_checks_pass':True,
    'token_space_bits':1000*math.log2(200000),
    'empirical_claim':False},indent=2)+'\n')
print('Scenario crossing checks passed; hypothetical plot generated.')
