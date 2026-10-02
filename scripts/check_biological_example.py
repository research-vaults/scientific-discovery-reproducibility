"""Exact two-world identification illustration; no agent or biological experiments."""
from pathlib import Path
from itertools import product
from collections import defaultdict
import json
P=Path(__file__).resolve().parents[1]
worlds=((1,0),(0,1))
def observation(world,setting):
    beta,gamma=world
    condition,laboratory=setting
    return beta*condition+gamma*laboratory

def optimum(schedule,target):
    groups=defaultdict(list)
    for w in worlds:
        transcript=tuple(observation(w,s) for s in schedule)
        groups[transcript].append(w)
    # Enumerate every deterministic decision rule on the observable classes.
    values=sorted({target(w) for w in worlds})
    return max(sum(sum(target(w)==answer for w in group)
                   for group,answer in zip(groups.values(),answers))/len(worlds)
               for answers in product(values,repeat=len(groups)))

rows=[]
for n in range(9):
    for schedule in product(((0,0),(1,1)),repeat=n):
        attribution=optimum(schedule,lambda w:w[0])
        separated=optimum(schedule+((1,0),),lambda w:w[0])
        prediction=optimum(schedule,lambda w:observation(w,(1,1)))
        assert attribution==0.5 and separated==1 and prediction==1
        rows.append(dict(n=n,schedule=schedule,attribution=attribution,
                         attribution_with_off_diagonal=separated,
                         prediction_on_diagonal=prediction))
assert len(rows)==511
out=dict(status='PASS',evidence_type='exact illustrative construction',worlds=worlds,
         prior=[0.5,0.5],schedules=511,conditions=2,
         attribution_configurations=1022,agent_runs=0,biological_experiments=0,
         conditional_attribution_ceiling=0.5,off_diagonal_attribution_ceiling=1,
         diagonal_prediction_ceiling=1,rows=rows)
(P/'evidence/biological_identification.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS: 511 schedules, two evidence conditions; exact attribution and prediction controls.')
