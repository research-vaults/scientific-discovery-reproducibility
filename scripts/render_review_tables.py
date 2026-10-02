"""Render all newly reported numerical summaries from retained outputs."""
from pathlib import Path
import json
P=Path(__file__).resolve().parents[1];O=P/'evidence/review_remediation_2026-09-19/results';F=P/'manuscript/figures'
def read(n):return json.loads((O/n).read_text())
def write(n,s):(F/n).write_text(s+'\n')
a=read('association_summary.json');h,g=a
write('association_main_text.tex',f"The complete-case panel contains {h['models']} models in {h['families']} published families. HumanEval's slope is {h['slope']:.2f} [{h['slope_cluster_percentile95'][0]:.2f}, {h['slope_cluster_percentile95'][1]:.2f}] log-odds per source SD; GSM8K's is {g['slope']:.2f} [{g['slope_cluster_percentile95'][0]:.2f}, {g['slope_cluster_percentile95'][1]:.2f}]. Held-out prediction uses the common {h['heldout_models']}-model panel with compute metadata, excluding each test model's family from fitting. On HumanEval, log compute predicts better than PC1: mean squared error against aggregate success rates (rate-MSE) is .0166 versus .0175. On GSM8K, three PCs predict better than log compute: .0084 versus .0348. Positive coupling thus need not make a capability index the best predictor.\n")
d=read('descriptor_summary.json');idx={(r['axis'],r['predictor']):r for r in d['results']}
write('descriptor_main_text.tex',f"The panel contains {d['models']} models from {d['model_families']} published families and {d['observed_cells']} model--task rates. Adding count and its interactions to PC1--3 reduces rate-MSE from .0355 to .0277 on held-out task families and from .0330 to .0238 on held-out model families (Table~\\ref{{tab:descriptor}}). Family-resampling intervals for these paired differences exclude zero on this archive; they do not estimate uncertainty over new task families. At seven objects, mean error also falls, but the interval for the error difference includes zero. Secondary controls test whether count alone or the known chance rate explains the gain.\n")
joint=json.loads((P/'evidence/joint_holdout_2026-09-19_151855/results/scores.json').read_text());ji={r['predictor']:r for r in joint if r['axis']=='joint_families' and r['task_family']=='all'}
rows=[r'\begin{tabular}{lrrrrr}',r'\toprule',r'& \multicolumn{3}{c}{Separate holdouts} & \multicolumn{2}{c}{Joint holdout}\\',r'Predictor & Task & Model & Size & Cells & Families\\',r'\midrule']
labels={'pooled_mean':'Pooled mean','task_family_mean':'Task-family mean','PC1':'PC1','PC3':'PC1--3','count_only':r'Count only$^\dagger$','PC3_count_additive':r'PC1--3 + count$^\dagger$','PC3_chance_floor':r'PC1--3 + chance floor$^\dagger$','PC3_count':'PC1--3 + count interactions'}
for m,l in labels.items():
 vals=[idx[axis,m]['aggregate_rate_mse'] for axis in ['target_family','model_family','larger_target']]+[ji[m][w]['aggregate_rate_mse'] for w in ['equal_cell','equal_family']]
 rows.append(l+' & '+' & '.join(f'{v:.4f}' for v in vals)+r'\\')
rows+= [r'\bottomrule',r'\end{tabular}'];write('descriptor_table.tex','\n'.join(rows))
r=read('identification.json');rr=[x for x in r if x['p0']==.05 and x['Delta']==2 and x['lambda']==1]
write('identification_main_text.tex',f"We enumerate success-count pairs at sample sizes 25/100/500, initial yields .01/.05/.2, separations .5/1/2 and couplings 0/.5/1. These 81 prespecified settings check coverage and recovery, including zero coupling and low yield. With initial target success .05, true coupling 1 and separation 2, the probability of a positive lower coupling bound is {100*rr[0]['positive_lower_probability']:.1f}\\% at $N=25$, {100*rr[1]['positive_lower_probability']:.1f}\\% at $N=100$, and above 99.9\\% at $N=500$ per level. At $N=100$, the median interval width is {rr[1]['median_width']:.2f} and {100*rr[1]['infinite_probability']:.2f}\\% of intervals remain unbounded. The finite grid's minimum coverage is {100*min(x['coverage'] for x in r):.2f}\\%; the analytical union-bound argument supplies the general guarantee (Appendix~\\ref{{app:identification}}).\n")
pil=read('pilot_extensions.json');r=next(x for x in pil if x['p1']==.2 and x['pair_dependence']=='independent' and x['design']=='sequential')
write('pilot_extension_main_text.tex',f"We also test stopping after 25, 50 or 100 observations per arm, splitting the error allowance equally across these checks. When candidate yield rises from .1 to .2, this rule costs {r['expected_pilot_cost']:.2f} units on average versus 2100 for the fixed pilot. Adoption falls from 4.572\\% to {100*r['adopt']:.3f}\\%. The rule spends almost the entire pilot budget on average. This is one specified design, not an optimal sequential policy.\n")
# Supplement: every holdout predictor with both loss functions.
rows=[r'\begin{tabular}{llrr}',r'\toprule',r'Holdout & Predictor & Rate-MSE & Cross-entropy\\',r'\midrule']
for r in d['results']:
 rows.append(r['axis'].replace('_',r'\_')+' & '+r['predictor'].replace('_',r'\_')+f" & {r['aggregate_rate_mse']:.5f} & {r['cross_entropy']:.5f}"+r'\\')
rows+=[r'\bottomrule',r'\end{tabular}'];write('descriptor_full_table.tex','\n'.join(rows))
rows=[r'\begin{tabular}{lrr}',r'\toprule',r'Target / predictor & Rate-MSE & Cross-entropy\\',r'\midrule']
for r in a:
 for m,v in r['heldout_scores'].items():rows.append(r['target']+' / '+m.replace('_',r'\_')+f" & {v['aggregate_rate_mse']:.5f} & {v['cross_entropy']:.5f}"+r'\\')
rows+=[r'\bottomrule',r'\end{tabular}'];write('association_full_table.tex','\n'.join(rows))
rows=[r'\begin{tabular}{rrrrrr}',r'\toprule',r'$N$ & $p_0$ & $\Delta$ & $\lambda$ & Exclude zero (\%) & Infinite (\%)\\',r'\midrule']
for r in read('identification.json'):
 if r['Delta']==2 and r['lambda']==1:
  rows.append(f"{r['N_per_level']} & {r['p0']:.2f} & 2 & 1 & {100*r['positive_lower_probability']:.3f} & {100*r['infinite_probability']:.3f}"+r'\\')
rows+=[r'\bottomrule',r'\end{tabular}'];write('identification_table.tex','\n'.join(rows))
rows=[r'\begin{tabular}{lrrrr}',r'\toprule',r'$p_1$ / design & Adopt (\%) & Non-saving (\%) & Unresolved (\%) & Mean cost\\',r'\midrule']
for r in pil:
 if r['pair_dependence']=='independent':rows.append(f"{r['p1']:.2f} / {r['design']} & {100*r['adopt']:.3f} & {100*r['non_saving']:.3f} & {100*r['unresolved']:.3f} & {r['expected_pilot_cost']:.2f}"+r'\\')
rows+=[r'\bottomrule',r'\end{tabular}'];write('sequential_table.tex','\n'.join(rows))
rows=[r'\begin{tabular}{rrrr}',r'\toprule',r'$N$ & Cost & EVSI per campaign & Break-even $K$\\',r'\midrule']
for r in read('value_of_information.json'):
 if r['K_campaigns']==1 and r['N_candidate_only']>0:
  gain=r['no_pilot_expected_reserve']-r['posterior_optimal_expected_reserve'];import math
  rows.append(f"{r['N_candidate_only']} & {r['pilot_cost']} & {gain:.3f} & {math.ceil(r['pilot_cost']/gain)}"+r'\\')
rows+=[r'\bottomrule',r'\end{tabular}'];write('voi_table.tex','\n'.join(rows))
write('descriptor_rival_text.tex',r"The additive count model has lower rate-MSE than the interaction model on all three separate holdout axes: rate-MSE .0260 versus .0277 on held-out task families, .0234 versus .0238 on held-out model families, and .0179 versus .0195 at seven objects. Count alone is worse than either capability-augmented count model. The chance-floor model also improves on capability-only prediction, and beats the interaction model on held-out task families. Thus these results support including simple task-size or chance information here, not the need for the richer interaction structure. The additive account has the lowest rate-MSE in these three comparisons; the joint holdouts below further challenge this choice.")
u={x['axis']:x for x in read('descriptor_uncertainty.json') if x['comparator']=='PC3'}
parts=[]
for ax,label in [('target_family','held-out task families'),('model_family','held-out model families'),('larger_target','seven objects')]:
 r=u[ax];ci=r['cluster_percentile95'];parts.append(f"{r['delta_mse']:.5f} [{ci[0]:.5f}, {ci[1]:.5f}] for {label}")
write('descriptor_uncertainty_text.tex',r"Paired differences resample the 20 published model families of fixed out-of-fold predictions, 2000 times. Against PC1--3, the primary count-interaction model's rate-MSE differences are "+'; '.join(parts)+'.')
print('Generated main text and data tables')
