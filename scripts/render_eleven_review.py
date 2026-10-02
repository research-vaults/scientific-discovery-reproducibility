"""Render retained review-remediation results without refitting."""
from pathlib import Path
import json,math,sys
P=Path(__file__).resolve().parents[1];E=P/'evidence/eleven_review_remediation_2026-09-21_001619/results';F=P/'manuscript/figures'
def read(n):return json.loads((E/n).read_text())
def table(name,header,rows,spec):
 (F/name).write_text('\n'.join([r'\begin{tabular}{'+spec+'}',r'\toprule',header+r'\\',r'\midrule']+[x+r'\\' for x in rows]+[r'\bottomrule',r'\end{tabular}'])+'\n')
s=read('descriptor_sensitivity_scores.json')
def score(axis,model,task='all'):return next(x for x in s if x['axis']==axis and x['predictor']==model and x['task_family']==task)['equal_cell']['aggregate_rate_mse']
rows=[]
for label,axis,task in [('Joint families','joint_families','all'),('Joint families + 7 objects','joint_families_larger','all'),('\\quad Logical deduction','joint_families_larger','logical_deduction'),('\\quad Object tracking','joint_families_larger','tracking_shuffled_objects')]:
 vals=[score(axis,m,task) for m in ['PC3','additive','chance_floor']];cells=next(x['cells'] for x in s if x['axis']==axis and x['task_family']==task and x['predictor']=='PC3');rows.append(label+f' & {cells} & '+' & '.join(f'{v:.4f}' for v in vals))
table('joint_main_comparison.tex','Holdout / task & Cells & PC1--3 & + count & Chance floor',rows,'lrrrr')
if '--joint-only' in sys.argv:
 print('joint table rendered from retained scores');sys.exit(0)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit

pil=json.loads((P/'evidence/decision_pilot/results.json').read_text());pil=pil['results'] if isinstance(pil,dict) and 'results' in pil else pil
rows=[]
for n,p1 in [(100,.1),(100,.12),(100,.2),(1000,.2)]:
 r=next(x['interval'] for x in read('decision_robustness.json') if x['interval']['N_per_arm']==n and x['interval']['p1']==p1)
 rows.append(f"{n} & {p1:.2f} & {100*r['interval_adopt']:.3f} & {100*r['interval_unresolved']:.3f}")
table('pilot_selected.tex',r'$N$ per arm & Candidate yield & Adopt (\%) & Unresolved (\%)',rows,'rrrr')
labels={'PC3':'PC1--3, logit','additive':'PC1--3 + count, logit','quadratic_PC3':'Quadratic capability','PC3_probit_0.005':'PC1--3, probit','additive_probit_0.005':'+ count, probit','PC3_rate':'PC1--3, rate fit','additive_rate':'+ count, rate fit','PC3_logit_0.001':'PC1--3, clip .001','additive_logit_0.001':'+ count, clip .001','PC3_logit_0.01':'PC1--3, clip .01','additive_logit_0.01':'+ count, clip .01','chance_floor':'Chance floor','chance_plus_count':'Chance floor + count','chance_feature_plus_count':'Chance feature + count'}
rows=[label+' & '+' & '.join(f'{score(ax,m):.4f}' for ax in ['task_family','model_family','larger','joint_families','joint_families_larger']) for m,label in labels.items()]
table('descriptor_sensitivity_table.tex','Predictor & Task out & Model out & Size out & Joint & Joint + size',rows,'lrrrrr')
rows=[]
for r in read('descriptor_refit_summary.json'):
 if r['metric']=='equal_cell_delta':
  label=('Joint' if r['axis']=='joint_families' else 'Joint + size')+' / '+{'all':'pooled','logical_deduction':'deduction','tracking_shuffled_objects':'tracking'}[r['task_family']]
  ci=r['percentiles_025_50_975'];rows.append(label+' & '+f'{ci[0]:.4f} & {ci[1]:.4f} & {ci[2]:.4f}')
table('descriptor_refit_table.tex','Contrast & 2.5th & Median & 97.5th percentile',rows,'lrrr')
rows=[]
for r in read('decision_robustness.json'):
 n=r['directed']['N_per_arm'];p=r['directed']['p1']
 for name in ['directed','sequential']:
  d=r[name];rows.append(f"{n} & {p:.2f} & {name} & {100*d['adopt']:.3f} & {100*d['non_saving']:.3f} & {100*d['unresolved']:.3f} & {d['pilot_cost']:.2f}")
table('directed_pilot_table.tex',r'$N$ & $p_1$ & Rule & Adopt (\%) & Non-saving (\%) & Defer (\%) & Cost',rows,'rrlrrrr')
rows=[]
for r in read('operational_policy.json'):
 rows.append(f"{r['prior']} & {r['N']} & {r['expected_implementable_reserve']:.2f} & {100*r['prior_predictive_future_success']:.2f} & {r['expected_future_expenditure']:.2f} & {100*r['pilot_at_least_one_hit']:.2f}")
table('operational_policy_table.tex',r'Prior & $N$ & Future reserve & Success (\%) & Expenditure & Pilot hit (\%)',rows,'lrrrrr')
f=pd.read_csv(E/'finite_table_family_scores.csv');rows=[]
for fam in f.family.unique():
 h=f[(f.family==fam)&(f.pilot==5)&(f.budget==20)&(f.stratum=='all')]
 vals=[h[h.predictor==m].iloc[0].brier for m in ['no_target','best','pilot_summary']]
 rows.append(fam.replace('-','--')+' & '+' & '.join(f'{v:.5f}' for v in vals))
table('finite_table_table.tex','Source family & Budget-only & + best gap & + pilot summaries',rows,'lrrr')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
x=np.linspace(0,10,501);fig,ax=plt.subplots(figsize=(6.4,2.65),layout='constrained')
for lam,col,ls in [(1,'#0072B2','-'),(.5,'#D55E00','--'),(0,'#666666',':')]:ax.plot(x,expit(-3+lam*x),color=col,ls=ls,label=f'Response slope {lam:g}')
ax.axhline(.8,color='black',lw=.8);ax.text(.12,.82,'Required success 0.8',fontsize=9)
for lam,col in [(1,'#0072B2'),(.5,'#D55E00')]:
 t=(np.log(4)+3)/lam;ax.plot([t,t],[0,.8],color=col,lw=.8,ls=':');ax.annotate(f'{t:.2f} years',(t,.8),xytext=(-15,-38),textcoords='offset points',fontsize=9)
ax.set(xlabel='Hypothetical elapsed years',ylabel='Target success probability',xlim=(0,10),ylim=(0,1));ax.legend(loc='lower right',frameon=False,fontsize=9)
fig.savefig(F/'central_horizons.pdf',metadata={'CreationDate':None,'ModDate':None});plt.close(fig)
print('eleven-review tables and central figure rendered')
