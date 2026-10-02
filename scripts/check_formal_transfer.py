"""Exact examples for the conditional reduction and resource-frontier argument.
These checks exercise illustrations, not empirical capabilities or theorem proofs.
"""
from pathlib import Path
from fractions import Fraction as F
import json, math
P=Path(__file__).resolve().parents[1]
k, b, a, delta, eps = 3, 10, 20, F(1,100), F(2,100)
failure_bound=k*delta+eps
assert 1-failure_bound==F(95,100) and a+k*b==50
# Disjoint failure events attain the union bound. Perfectly correlated call
# failures satisfy the same bound without attaining it; no independence used.
assert 3*delta+eps==F(5,100)
assert delta+eps<=failure_bound
k2,a2,eps2=2,7,F(1,100)
assert a2+k2*(a+k*b)==107
assert eps2+k2*(k*delta+eps)==F(11,100)
# High source-average success need not survive target query selection.
source_success=F(99,100)*1+F(1,100)*0
target_success=F(0)*1+F(1)*0
assert source_success==F(99,100) and target_success==0
# Two feasible resource upper sets are incomparable: each generator is a witness.
g1,g2=(10,10),(100,1)
feasible=lambda budget,g:all(x>=y for x,y in zip(budget,g))
assert feasible(g1,g1) and not feasible(g1,g2)
assert feasible(g2,g2) and not feasible(g2,g1)
# Exact special-case frontier and independent finite-difference slope check.
alpha,beta,eta=F(2),F(3,10),F(1,10)
assert (beta+eta)/alpha==F(1,5)
R0,B0=2.,4.
n=lambda t:(math.log(B0/R0)+float(beta+eta)*t)/float(alpha)
for t in (0.,1.,2.,3.):
 assert abs(R0*math.exp(float(alpha)*n(t)-float(eta)*t)-B0*math.exp(float(beta)*t))<1e-10
 h=1e-5
 assert abs((n(t+h)-n(t-h))/(2*h)-float((beta+eta)/alpha))<1e-8
out={'evidence_kind':'exact hypothetical illustrations, not empirical estimates','reduction':{'calls':k,'call_cost':b,'overhead':a,'call_error':float(delta),'residual_error':float(eps),'target_budget_upper_bound':a+k*b,'target_success_lower_bound':float(1-failure_bound)},'composition':{'target_budget_upper_bound':107,'failure_upper_bound':.11},'distribution_counterexample':{'source_success':float(source_success),'target_success':float(target_success)},'resource_order':{'generators':[g1,g2],'incomparable':True},'frontier_special_case':{'alpha':float(alpha),'beta':float(beta),'eta':float(eta),'slope':.2,'parameters_stipulated':True},'status':'PASS'}
p=P/'evidence/integration_audit_2026-09-15/formal_checks.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(out,indent=2)+'\n');print('Formal transfer/frontier illustration checks PASS')
