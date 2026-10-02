"""Check stipulated pricing and dependence examples; no measured AI outcomes."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import json
P=Path(__file__).resolve().parents[1]
old_cost=60+40
new_cost=F(60,10)+40
assert old_cost==100 and new_cost==46
# Enumerate the full joint distribution of ten independent Bernoulli trials,
# rather than merely re-evaluating the displayed closed-form expression.
p=F(1,5)
success=F(0)
for seq in product([0,1],repeat=10):
 mass=F(1)
 for x in seq: mass*=p if x else 1-p
 if any(seq): success+=mass
assert success==1-(1-p)**10
correlated=p
assert correlated<success
# Finite witness: only procedure A qualifies at the specified reliability.
# Price falls can make it affordable; a hard deadline still blocks it.
procedures=[{'name':'A','resources':(60,40),'time':20,'reliability':F(9,10)},
            {'name':'B','resources':(10,20),'time':5,'reliability':F(3,5)}]
def attainable(prices,budget,deadline):
 return {a['name'] for a in procedures if a['reliability']>=F(9,10)
         and a['time']<=deadline
         and sum(x*y for x,y in zip(prices,a['resources']))<=budget}
assert attainable((1,1),50,30)==set()
assert attainable((F(1,10),1),50,30)=={'A'}
assert attainable((F(1,10),1),50,10)==set()
for B in [0,20,30,46,50,100]:
 for T in [5,10,20,30]:
  assert attainable((1,1),B,T)<=attainable((F(1,10),1),B,T)
out={'status':'PASS','evidence_kind':'stipulated examples, not empirical results',
 'cost':{'before':100,'after_compute_price_divided_by_ten':46,'speedup_of_affordability_fixed_procedure':float(F(100,46))},
 'ten_complete_attempts':{'marginal_success':.2,'independent_success':float(success),'perfectly_correlated_success':float(correlated),'assumes_correct_recognition':True},
 'finite_affordability_witness':{'price_cut_can_expand_reach':True,'unchanged_deadline_can_still_block':True,'enumerated_inclusion_checks':24}}
f=P/'evidence/agents_prices_2026-09-15/exact_checks.json';f.parent.mkdir(exist_ok=True);f.write_text(json.dumps(out,indent=2)+'\n')
print('Agent-price/dependence illustration checks PASS')
