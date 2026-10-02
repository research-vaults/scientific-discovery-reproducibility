"""Render measured main contrast and retain complete endpoint sensitivity tables."""
from pathlib import Path
import csv,json,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];E=P/'evidence/consequential_gap_repairs_2026-09-26_012131';R=E/'results';F=P/'manuscript/figures'
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'font.family':'DejaVu Sans'})
scores=json.loads((P/'evidence/joint_holdout_2026-09-19_151855/results/scores.json').read_text())
boot=json.loads((P/'evidence/eleven_review_remediation_2026-09-21_001619/results/descriptor_refit_summary.json').read_text())
ix={(r['axis'],r['predictor'],r['task_family']):r for r in scores}
bi={(r['axis'],r['task_family'],r['metric']):r for r in boot}
rows=[('joint_families','all','Joint families (438 cells)'),('joint_families_larger','all','Joint + larger size (146)'),('joint_families_larger','logical_deduction','Larger deduction (73)'),('joint_families_larger','tracking_shuffled_objects','Larger tracking (73)')]
fig,ax=plt.subplots(figsize=(6.2,2.05));manifest=[]
for i,(axis,task,label) in enumerate(rows):
 d=ix[axis,'PC3_count_additive',task]['equal_cell']['aggregate_rate_mse']-ix[axis,'PC3',task]['equal_cell']['aggregate_rate_mse']
 lo,med,hi=bi[axis,task,'equal_cell_delta']['percentiles_025_50_975']; y=3-i
 ax.plot([lo,hi],[y,y],color='#246391',lw=2);ax.plot([lo,lo],[y-.08,y+.08],color='#246391');ax.plot([hi,hi],[y-.08,y+.08],color='#246391');ax.scatter([d],[y],s=28,color='#ad4527' if d>0 else '#246391',zorder=4)
 manifest.append(dict(axis=axis,task=task,delta=d,percentile_low=lo,percentile_high=hi))
ax.axvline(0,color='0.4',lw=1,ls='--');ax.set_yticks(range(4),[r[2] for r in rows][::-1]);ax.set_xlabel('Change in rate-MSE: count-informed minus capability-only');ax.set_xlim(-.065,.019);ax.set_ylim(-.5,3.5);ax.spines[['top','right']].set_visible(False);ax.text(.02,.98,'Lower favors count information',transform=ax.transAxes,va='top',fontsize=8,color='#246391');fig.tight_layout(pad=.6);fig.savefig(F/'joint_effects_main.pdf');plt.close(fig)
(R/'plot_values.json').write_text(json.dumps(manifest,indent=2))
summary=list(csv.DictReader((R/'summary.csv').open()));si={(float(r['quantile']),int(r['pilot']),int(r['budget']),r['predictor'],r['stratum']):r for r in summary}
lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Percentile & Budget & Hit rate & Budget-only & Best gap & Summaries\\',r'\midrule']
for q in [.9,.95,.99]:
 for budget in [20,40]:
  a=si[q,5,budget,'no_target','all'];b=si[q,5,budget,'best','all'];c=si[q,5,budget,'pilot_summary','all']
  lines.append(f'{int(q*100)} & {budget} & {100*float(a["hit_rate"]):.1f}'+r'\% & '+f'{float(a["brier"]):.5f} & {float(b["brier"]):.5f} & {float(c["brier"]):.5f}'+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(F/'endpoint_main.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Percentile & Pilot/budget & Budget & Best gap & Summaries & Prevalence\\',r'\midrule']
for st in ['all','not_yet_hit']:
 lines.append(r'\multicolumn{6}{l}{'+('All campaigns' if st=='all' else 'No pilot hit')+r'}\\')
 for q in [.9,.95,.99]:
  for pilot in [5,10]:
   for budget in [20,40]:
    vals=[float(si[q,pilot,budget,name,st]['brier']) for name in ['no_target','best','pilot_summary','prevalence']]
    lines.append(f'{int(q*100)} & {pilot}/{budget} & '+' & '.join(f'{v:.5f}' for v in vals)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(F/'endpoint_full.tex').write_text('\n'.join(lines)+'\n')
pair=list(csv.DictReader((R/'paired_family_differences.csv').open()));pi={(float(r['quantile']),r['family']):r for r in pair if r['pilot']=='5' and r['budget']=='20' and r['stratum']=='all'}
lines=[r'\begin{tabular}{lrrr}',r'\toprule',r'Source & 90th & 95th & 99th\\',r'\midrule']
for family,label in [('additives','Additive screens'),('buchwald-hartwig','Buchwald--Hartwig'),('c2-yield','C2-yield'),('suzuki-miyaura','Suzuki--Miyaura')]:
 lines.append(label+' & '+' & '.join(f'{float(pi[q,family]["brier_delta"]):+.5f}' for q in [.9,.95,.99])+r'\\')
lines += [r'\bottomrule',r'\end{tabular}'];(F/'endpoint_family.tex').write_text('\n'.join(lines)+'\n')
q=.8; a=math.log(q/(1-q))+3;b=2*a;dA=4;dB=.25;T=10
na=math.floor((T-a)/dA);nb=math.floor((T-b)/dB)
assert (na,nb)==(1,4)
case=dict(q=q,availability_A=a,availability_B=b,duration_A=dA,duration_B=dB,deadline=T,completed_attempts_A=na,completed_attempts_B=nb,success_A=1-(1-q)**na,success_B=1-(1-q)**nb,mean_A=a+dA/q,mean_B=b+dB/q,boundary=q*(b-a),duration_gap=dA-dB)
assert case['duration_gap']>case['boundary'];assert case['mean_B']<case['mean_A']
(R/'worked_decision.json').write_text(json.dumps(case,indent=2));print(case)
