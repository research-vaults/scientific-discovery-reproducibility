"""Exact, benign finite-law examples; no LLM or biological experiments."""
from pathlib import Path
from collections import Counter
import csv, json, random
P=Path(__file__).resolve().parents[1]
def rank(rows):
 basis={}
 for x in rows:
  while x:
   k=x.bit_length()-1
   if k in basis:x^=basis[k]
   else:basis[k]=x;break
 return len(basis)
def transcript(theta,rows):return tuple((theta & x).bit_count()%2 for x in rows)
def verify(d,rows):
 counts=Counter(transcript(t,rows) for t in range(1<<d))
 r=rank(rows);p=len(counts)/(1<<d)
 assert len(counts)==1<<r and set(counts.values())=={1<<(d-r)}
 assert p==2**(r-d)
 return r,len(counts),p
rows=[]
for d in range(1,9):
 basis=[1<<i for i in range(d)]
 configs={'one_observation':([1],[]),'duplicates_in_one_request':([1]*d,[]),'independent_batch':(basis,[]),'complementary_prior':([1],basis[1:]),'redundant_prior':([1],basis[:1])}
 for name,(obs,prior) in configs.items():
  r,n,p=verify(d,obs+prior)
  rows.append(dict(d=d,condition=name,requests=1,observations=len(obs),prior_constraints=len(prior),joint_rank=r,transcript_classes=n,hidden_laws=1<<d,bayes_exact_recovery=p))
rng=random.Random(20260915)
checks=0
for d in range(1,9):
 for j in range(30):
  matrix=[rng.randrange(1<<d) for _ in range(rng.randrange(0,2*d+1))];verify(d,matrix);checks+=1
# Positive/negative target control: theta1 is identifiable; theta2 is not from theta1.
for d in range(2,9):
 for q,expected in [(1,1.0),(2,0.5)]:
  bins={}
  for t in range(1<<d):bins.setdefault(transcript(t,[1]),Counter())[(t&q).bit_count()%2]+=1
  assert sum(max(c.values()) for c in bins.values())/(1<<d)==expected
with (P/'evidence/finite_examples.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(P/'evidence/finite_examples_checks.json').write_text(json.dumps({'status':'PASS','protocol':'C1','enumerated_designs':len(rows),'additional_random_matrix_checks':checks,'target_controls':14,'dimension_range':[1,8],'random_seed':20260915,'LLM_runs':0,'scope':'Exact arithmetic illustration, not empirical model performance'},indent=2))
print(json.dumps({'designs':len(rows),'random_checks':checks,'d8':rows[-5:]},indent=2))
