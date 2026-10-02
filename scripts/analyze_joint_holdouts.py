"""Post-result challenge under the frozen joint-holdout follow-up contract."""
from pathlib import Path
import ast, json, hashlib
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import norm

P = Path(__file__).resolve().parents[1]
E = P / 'evidence/joint_holdout_2026-09-19_151855'
O = E / 'results'
O.mkdir(parents=True, exist_ok=True)
S = P / 'evidence/review_remediation_2026-09-19/sources/ObsScaling/eval_results'
# Execute only the three existing pure numerical helpers, not the earlier study.
# This preserves identical transformations and estimator definitions.
tree = ast.parse((P / 'scripts/analyze_review_evidence.py').read_text())
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)
           and n.name in {'pcs', 'regress', 'metrics'}]
assert len(helpers) == 3
exec(compile(ast.Module(body=helpers, type_ignores=[]), 'shared_numerical_helpers', 'exec'))
src = ['MMLU', 'ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd']
base = pd.read_csv(S / 'base_llm_benchmark_eval.csv').dropna(subset=src + ['Model Family'])
post = pd.read_csv(S / 'base_llm_post_training_eval.csv')
frames = []
for task in ['logical_deduction', 'tracking_shuffled_objects']:
    for n, word in [(3, 'three'), (5, 'five'), (7, 'seven')]:
        col = f'bbh_cot_fewshot_{task}_{word}_objects_3_exact_match,strict-match'
        m = base.merge(post[['Model', col]], on='Model', validate='one_to_one').dropna(subset=[col])
        frames.append(m[['Model', 'Model Family'] + src].assign(task_family=task, n=n, y=m[col]))
a = pd.concat(frames, ignore_index=True)
assert len(a) == 438 and a.Model.nunique() == 73 and a['Model Family'].nunique() == 20
assert not a.duplicated(['Model', 'task_family', 'n']).any()

def predict(tr, te):
    unique_sources = tr.drop_duplicates('Model')[src].to_numpy()
    _, x = pcs(unique_sources, tr[src].to_numpy())
    _, xt = pcs(unique_sources, te[src].to_numpy())
    c = np.log(tr.n.to_numpy()); ct = np.log(te.n.to_numpy())
    center, scale = c.mean(), c.std()
    c = (c-center)/scale; ct = (ct-center)/scale
    designs = {'PC1': (x[:, :1], xt[:, :1]), 'PC3': (x, xt),
               'count_only': (c[:, None], ct[:, None]),
               'PC3_count_additive': (np.column_stack([x, c]), np.column_stack([xt, ct])),
               'PC3_count': (np.column_stack([x, c, x*c[:, None]]),
                             np.column_stack([xt, ct, xt*ct[:, None]]))}
    out = {name: regress(v[0], tr.y.to_numpy(), v[1], penalty=1)[0]
           for name, v in designs.items()}
    out['pooled_mean'] = np.full(len(te), tr.y.mean())
    out['task_family_mean'] = np.array([tr.loc[tr.task_family == t, 'y'].mean()
                                      if (tr.task_family == t).any() else tr.y.mean()
                                      for t in te.task_family])
    floor = 1/tr.n.to_numpy(); floor_test = 1/te.n.to_numpy()
    relative = (tr.y.to_numpy()-floor)/(1-floor)
    out['PC3_chance_floor'] = floor_test+(1-floor_test)*regress(x, relative, xt, penalty=1)[0]
    return out

predictions, folds = [], []
label_invariance_checks = 0
for axis in ['joint_families', 'joint_families_larger']:
    for family in sorted(a['Model Family'].unique()):
        for task in sorted(a.task_family.unique()):
            train = (a['Model Family'] != family) & (a.task_family != task)
            test = (a['Model Family'] == family) & (a.task_family == task)
            if axis == 'joint_families_larger':
                train &= a.n < 7
                test &= a.n == 7
            tr, te = a[train].copy(), a[test].copy()
            assert len(tr) and len(te)
            assert not set(tr.Model) & set(te.Model)
            assert not set(tr['Model Family']) & set(te['Model Family'])
            assert not set(tr.task_family) & set(te.task_family)
            if axis == 'joint_families_larger':
                assert set(tr.n) == {3, 5} and set(te.n) == {7}
            pp = predict(tr, te)
            altered = te.copy(); altered['y'] = 1-altered.y
            pp_altered = predict(tr, altered)
            for name, values in pp.items():
                assert np.array_equal(values, pp_altered[name])
                assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
                label_invariance_checks += 1
                for (_, row), p in zip(te.iterrows(), values):
                    predictions.append(dict(axis=axis, fold=f'{family}|{task}', model=row.Model,
                                            model_family=row['Model Family'], task_family=task,
                                            objects=int(row.n), predictor=name,
                                            observed=float(row.y), predicted=float(p)))
            folds.append(dict(axis=axis, heldout_model_family=family, heldout_task_family=task,
                              train_rows=len(tr), train_models=tr.Model.nunique(),
                              train_model_families=tr['Model Family'].nunique(),
                              train_objects=sorted(tr.n.unique().tolist()), test_rows=len(te)))
df = pd.DataFrame(predictions)
for axis, n in [('joint_families', 438), ('joint_families_larger', 146)]:
    part = df[df.axis == axis]
    assert (part.groupby('predictor').size() == n).all()
    assert not part.duplicated(['predictor', 'model', 'task_family', 'objects']).any()
df.to_csv(O / 'predictions.csv', index=False)
old = pd.read_csv(P / 'evidence/review_remediation_2026-09-19/results/descriptor_predictions.csv')
combined = pd.concat([old, df], ignore_index=True)
summaries = []
for (axis, predictor), g in combined.groupby(['axis', 'predictor']):
    for task in ['all'] + sorted(g.task_family.unique()):
        h = g if task == 'all' else g[g.task_family == task]
        family_scores = [metrics(f.observed.to_numpy(), f.predicted.to_numpy())
                         for _, f in h.groupby('model_family')]
        summaries.append(dict(axis=axis, predictor=predictor, task_family=task,
                              cells=len(h), model_families=h.model_family.nunique(),
                              equal_cell=metrics(h.observed.to_numpy(), h.predicted.to_numpy()),
                              equal_family={k: float(np.mean([f[k] for f in family_scores]))
                                            for k in family_scores[0]}))
def save(name, value):
    (O / name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
save('scores.json', summaries)
save('folds.json', folds)
save('checks.json', dict(status='PASS', folds=len(folds), target_label_invariance_checks=label_invariance_checks,
                         test_cell_counts={'joint_families':438, 'joint_families_larger':146},
                         train_test_model_and_task_families_disjoint=True,
                         duplicate_test_cells=False, no_predictor_tuning=True,
                         scope='Retrospective follow-up on the same archive; not independent confirmation.'))
for axis in ['joint_families', 'joint_families_larger']:
    print(axis)
    for r in summaries:
        if r['axis'] == axis and r['task_family'] == 'all':
            print(r['predictor'], r['equal_cell'], r['equal_family'])
