"""Generate follow-up values directly from retained joint-holdout results."""
from pathlib import Path
import json
P = Path(__file__).resolve().parents[1]
O = P / 'evidence/joint_holdout_2026-09-19_151855/results'
F = P / 'manuscript/figures'
scores = json.loads((O / 'scores.json').read_text())
ix = {(r['axis'], r['predictor'], r['task_family']): r for r in scores}
def row(axis, predictor, task='all'):
    return ix[axis, predictor, task]
def mse(axis, predictor, weighting='equal_cell', task='all'):
    return row(axis, predictor, task)[weighting]['aggregate_rate_mse']
def write(name, value):
    (F / name).write_text(value+'\n')
axes = ['joint_families', 'joint_families_larger']
same = all((mse(a, 'PC3_count_additive') < mse(a, 'PC3')) ==
           (mse(a, 'PC3_count_additive', 'equal_family') < mse(a, 'PC3', 'equal_family'))
           for a in axes)
assert same, 'Inspect weighting reversal before choosing the main-text wording.'
write('joint_main_text.tex',
      'We then test jointly unseen model and task families in a follow-up chosen after the initial results. '
      'Adding count to PC1--3 lowers rate-MSE from '
      f"{mse('joint_families','PC3'):.4f} to {mse('joint_families','PC3_count_additive'):.4f}. "
      'When the same holdout also requires extrapolation to seven objects, error falls from '
      f"{mse('joint_families_larger','PC3'):.4f} to {mse('joint_families_larger','PC3_count_additive'):.4f}. "
      'Both average gains persist when model families receive equal weight '
      r'(Appendix~\ref{app:joint}). '
      'But seven-object logical deduction worsens from '
      f"{mse('joint_families_larger','PC3',task='logical_deduction'):.4f} to "
      f"{mse('joint_families_larger','PC3_count_additive',task='logical_deduction'):.4f}; "
      'shuffled-object tracking drives the pooled gain.')
labels = {'pooled_mean':'Pooled mean', 'task_family_mean':'Task-family mean', 'PC1':'PC1',
          'PC3':'PC1--3', 'count_only':'Count only', 'PC3_count_additive':'PC1--3 + count',
          'PC3_count':'PC1--3 + count interactions', 'PC3_chance_floor':'PC1--3 + chance floor'}
lines = [r'\begin{tabular}{lrrrr}', r'\toprule',
         r'& \multicolumn{2}{c}{Rate-MSE} & \multicolumn{2}{c}{Cross-entropy}\\',
         r'Predictor & Equal cell & Equal family & Equal cell & Equal family\\', r'\midrule']
for axis, label in zip(axes, ['Joint model/task holdout (438 cells)', 'Joint model/task holdout, seven objects (146 cells)']):
    lines.append(r'\multicolumn{5}{l}{'+label+r'}\\')
    for predictor, name in labels.items():
        r = row(axis, predictor)
        vals = [r[w][metric] for metric in ['aggregate_rate_mse','cross_entropy']
                for w in ['equal_cell','equal_family']]
        lines.append(name+' & '+' & '.join(f'{v:.5f}' for v in vals)+r'\\')
    lines.append(r'\midrule')
lines[-1] = r'\bottomrule'
lines.append(r'\end{tabular}')
write('joint_table.tex', '\n'.join(lines))
details = []
for axis, label in zip(axes, ['joint family holdout', 'joint family and size holdout']):
    ds = [mse(axis,'PC3_count_additive',task=t)-mse(axis,'PC3',task=t)
          for t in ['logical_deduction','tracking_shuffled_objects']]
    dfs = [mse(axis,'PC3_count_additive','equal_family',t)-mse(axis,'PC3','equal_family',t)
           for t in ['logical_deduction','tracking_shuffled_objects']]
    details.append(f'For {label}, additive-count minus PC1--3 rate-MSE is {ds[0]:+.5f} '
                   f'for logical deduction and {ds[1]:+.5f} for shuffled-object tracking; '
                   f'equal-family differences are {dfs[0]:+.5f} and {dfs[1]:+.5f}')
write('joint_subgroup_text.tex', '. '.join(details)+'. Negative differences favour the count descriptor. '
      'All per-task scores for the eight predictors and all five split axes are retained, including '
      'equal-family rescoring of the three earlier tests. These contrasts do not estimate uncertainty '
      'over a population of new task families.')
print('Generated joint-holdout table and result text')
