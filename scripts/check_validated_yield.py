"""Check the explicitly hypothetical campaign calculation; no empirical estimates."""
from pathlib import Path
import json, math
P=Path(__file__).resolve().parents[1]
def requirement(p,q):
    assert 0 < p < 1 and 0 < q < 1
    n=math.ceil(math.log1p(-q)/math.log1p(-p))
    # Check the actual threshold, not just another expression of the formula.
    assert 1-(1-p)**n >= q-1e-12
    assert 1-(1-p)**(n-1) < q+1e-12
    return n
rows=[]
for label,c,p in [('baseline',1,.1),('cheaper_proposals',.1,.1),('better_selection',2,.2),('costlier_unchanged_yield',2,.1)]:
    n=requirement(p,.9);rows.append(dict(procedure=label,c=c,v=9,p=p,q=.9,attempts=n,budget=n*(c+9),success_probability=1-(1-p)**n,success_one_fewer=1-(1-p)**(n-1)))
assert [r['attempts'] for r in rows]==[22,22,11,22]
assert all(math.isclose(a['budget'],b) for a,b in zip(rows,[220,200.2,121,242]))
assert math.isclose(1-rows[1]['budget']/220,.09)
assert math.isclose(1-rows[2]['budget']/220,.45)
sensitivity=[dict(q=q,baseline_attempts=requirement(.1,q),better_attempts=requirement(.2,q),baseline_budget=10*requirement(.1,q),better_budget=11*requirement(.2,q)) for q in [.5,.9,.99]]
# At cost 11, a strict saving over budget 220 permits at most 19 attempts.
p_threshold=1-.1**(1/19)
assert math.isclose(1-(1-p_threshold)**19,.9)
assert 1-(1-(p_threshold-1e-5))**19 < .9
# Complete positive dependence: all successes share one Bernoulli(.2) variable.
# Repetition never increases its .2 campaign success probability.
correlation=dict(p=.2,attempts=11,independent_success=1-.8**11,identical_outcome_success=.2)
# Exact decision reversals and monotone uncertainty propagation, not data fitting.
baseline_worlds=[dict(p=p,budget=10*requirement(p,.9)) for p in [.1,.3]]
assert [x['budget'] for x in baseline_worlds]==[220,70]
assert 121 < baseline_worlds[0]['budget'] and 121 > baseline_worlds[1]['budget']
intervals=[]
for lo,hi,expected in [(.12,.22,(110,209)),(.08,.22,(110,308))]:
    lower=11*requirement(hi,.9);upper=11*requirement(lo,.9)
    assert (lower,upper)==expected
    for i in range(101):
        prob=lo+(hi-lo)*i/100
        assert lower <= 11*requirement(prob,.9) <= upper
    intervals.append(dict(stipulated_p_range=[lo,hi],budget_range=[lower,upper]))
# Robust adoption versus abstention when both procedures have uncertain yield.
baseline_range=(10*requirement(.11,.9),10*requirement(.09,.9))
new_narrow=(11*requirement(.22,.9),11*requirement(.13,.9))
new_wide=(11*requirement(.22,.9),11*requirement(.08,.9))
assert baseline_range==(200,250) and new_narrow==(110,187) and new_wide==(110,308)
assert new_narrow[1] < baseline_range[0]
assert new_wide[0] < baseline_range[0] and new_wide[1] > baseline_range[1]
robust_decision=dict(baseline_budget_range=baseline_range,new_narrow_budget_range=new_narrow,new_wide_budget_range=new_wide,narrow_decision='robust saving under stipulated ranges',wide_decision='unresolved; acquire evidence')
# Non-identical conditional yields still obey the stated chain-rule bound.
conditional_yields=[.12,.19,.15,.25]
assert math.prod(1-x for x in conditional_yields) <= (1-min(conditional_yields))**len(conditional_yields)
# Prespecified setup sensitivity: reliability is per campaign, not joint across K.
setup_rows=[]
for S,K in [(0,1),(120,1),(120,10)]:
    cap=math.ceil((220-S/K)/11)-1
    threshold=1-.1**(1/cap)
    total_new=S+K*121;total_base=K*220
    assert S+K*11*cap < total_base
    assert S+K*11*(cap+1) >= total_base
    assert math.isclose(1-(1-threshold)**cap,.9,abs_tol=1e-12)
    assert 1-(1-(threshold-1e-6))**cap < .9
    setup_rows.append(dict(setup=S,campaigns=K,new_total=total_new,baseline_total=total_base,new_average=total_new/K,max_attempts_for_saving=cap,minimum_yield=threshold,saving=total_new<total_base))
assert [x['max_attempts_for_saving'] for x in setup_rows]==[19,9,18]
assert [x['saving'] for x in setup_rows]==[True,False,True]

result=dict(status='PASS',type='hypothetical deduction, not data',setup_sensitivity=setup_rows,rows=rows,sensitivity=sensitivity,minimum_yield_for_strict_saving_at_c2=p_threshold,correlated_counterexample=correlation,opposite_baseline_worlds=baseline_worlds,stipulated_uncertainty_ranges=intervals,uncertainty_aware_decision=robust_decision)
out=P/'evidence/validated_yield';out.mkdir(exist_ok=True)
(out/'checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
