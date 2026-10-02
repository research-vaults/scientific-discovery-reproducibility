"""Check exact hypothetical examples; no empirical validation is implied."""
from pathlib import Path
from itertools import permutations, combinations
from fractions import Fraction
from math import ceil, log
import json
P=Path(__file__).resolve().parents[1]
rows=[]
for bits,expected in [(64,948),(16,282)]:
    log_size=bits*log(2)
    b=ceil((log_size+log(20))/.05)
    assert b==expected
    assert log_size-b*.05 <= log(.05)
    assert log_size-(b-1)*.05 > log(.05)
    rows.append(dict(log2_class_size=bits,sufficient_budget=b))
# Exhaust all deterministic fresh-query orders for small N. Until the hit,
# every adaptive policy sees only negatives and therefore follows such an order.
for n in range(3,8):
    h=[tuple(int(i==theta) for i in range(n)) for theta in range(n)]
    assert all({v[i] for v in h}=={0,1} for i in range(n))
    assert all(len({(v[i],v[j]) for v in h})<4 for i,j in combinations(range(n),2))
    for order in permutations(range(n)):
        positions=[order.index(theta)+1 for theta in range(n)]
        assert Fraction(sum(positions),n)==Fraction(n+1,2)
        for k in range(n+1):
            assert Fraction(sum(t<=k for t in positions),n)==Fraction(k,n)
assert Fraction(999,1000)==1-Fraction(1,1000)
assert Fraction(1001,2)==Fraction('500.5')
# Boundary witness: removing the true rule can make a tiny class always wrong.
true=(1,1); restricted=[(0,0)]
assert min(sum(a!=b for a,b in zip(h,true))/2 for h in restricted)==1
weights_results=[]
for w in [Fraction(4,5),Fraction(1,5)]:
    qr=w;qs=1-w
    weights_results.append(dict(weight_first=float(w),distance_QR=float(qr),distance_QS=float(qs)))
assert weights_results[0]['distance_QR'] > weights_results[0]['distance_QS']
assert weights_results[1]['distance_QR'] < weights_results[1]['distance_QS']
# Nonnegative weights induce only positive-coordinate agreement at distance zero.
def response_distance(a,b,w):
    assert len(a)==len(b)==len(w) and all(x>=0 for x in w) and abs(sum(w)-1)<1e-12
    return sum(x*abs(y-z) for y,z,x in zip(a,b,w))
assert response_distance((0,0),(0,1),(1,0))==0
assert response_distance((0,0),(0,1),(.5,.5))==.5
zero_weight_check=dict(weights=[1,0],first=[0,0],second=[0,1],distance=0,interpretation='Zero distance does not imply agreement on ignored coordinates')
out=dict(status='PASS',finite_class=rows,response_weight_sensitivity=weights_results,zero_weight_boundary=zero_weight_check,rare_positive=dict(N=1000,accuracy=.999,expected_queries=500.5),enumerated_N=list(range(3,8)),scope='Constructed arithmetic and finite checks; not a proof of universal scientific scaling or an empirical result.')
f=P/'evidence/theory_landscape_2026-09-16/theory_checks.json'
f.parent.mkdir(parents=True,exist_ok=True)
f.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
