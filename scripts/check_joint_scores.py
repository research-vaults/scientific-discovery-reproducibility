"""Independent standard-library reconstruction of both loss weightings."""
from pathlib import Path
import csv, json, math, collections
P = Path(__file__).resolve().parents[1]
O = P / 'evidence/joint_holdout_2026-09-19_151855/results'
files = [P / 'evidence/review_remediation_2026-09-19/results/descriptor_predictions.csv',
         O / 'predictions.csv']
groups = collections.defaultdict(list)
for path in files:
    with path.open() as f:
        for row in csv.DictReader(f):
            y, p = float(row['observed']), float(row['predicted'])
            p = min(1-1e-8, max(1e-8, p))
            loss = ((y-p)**2, -y*math.log(p)-(1-y)*math.log1p(-p))
            for task in ['all', row['task_family']]:
                groups[(row['axis'], row['predictor'], task)].append((row['model_family'], loss))
maximum = 0.0
checks = 0
for record in json.loads((O / 'scores.json').read_text()):
    key = (record['axis'], record['predictor'], record['task_family'])
    values = groups[key]
    assert len(values) == record['cells']
    by_family = collections.defaultdict(list)
    for family, loss in values:
        by_family[family].append(loss)
    assert len(by_family) == record['model_families']
    for index, metric in enumerate(['aggregate_rate_mse', 'cross_entropy']):
        cell_score = math.fsum(v[index] for _, v in values)/len(values)
        family_score = math.fsum(math.fsum(v[index] for v in part)/len(part)
                                 for part in by_family.values())/len(by_family)
        for weight, score in [('equal_cell', cell_score), ('equal_family', family_score)]:
            difference = abs(score-record[weight][metric])
            assert difference < 1e-12, (key, weight, metric, difference)
            maximum = max(maximum, difference)
            checks += 1
result = dict(status='PASS', reconstructed_score_cells=checks, max_abs_difference=maximum,
              method='csv/math.fsum reconstruction, separate from pandas/numpy analysis')
(O / 'independent_score_checks.json').write_text(json.dumps(result, indent=2)+'\n')
print(result)
