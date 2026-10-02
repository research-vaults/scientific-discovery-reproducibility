"""Exact conceptual workflows; no empirical data or fitted parameters."""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1]
assert F(4,5)*F(3,4)+F(1,5)*F(1,4)==F(13,20)
assert F(13,20)-F(1,2)==F(3,20)
assert F(90,10)+10==19
assert F(10,10)+90==91
assert F(90,9)==10
assert F(90,90+10)==F(9,10)
assert F(90,90+10*9)==F(1,2)
x=np.geomspace(1,100,400)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(6.6,2.35),layout='constrained')
ax=axs[0]
ax.plot(x,90/x+10,label='A: 90/x + 10',color='#176b91',lw=2)
ax.plot(x,10/x+90,label='B: 10/x + 90',color='#c16a21',lw=2)
ax.axvline(9,color='#666666',ls=':',lw=1)
ax.set(xscale='log',xlabel='First-stage efficiency multiplier x',ylabel='Total duration (time units)',ylim=(0,106),title='(a) Same starting burden')
ax.legend(frameon=False,loc='center right',fontsize=8)
ax.text(1.8,69,'Stage tie\nin A: x = 9',fontsize=8,color='#444444')
ax=axs[1]
ax.plot(x,90/x+10,label='Only first stage improves',color='#176b91',lw=2)
ax.plot(x,100/x,label='Both stages improve (y = x)',color='#7e4587',lw=2,ls='--')
ax.axhline(10,color='#888888',lw=1,ls=':')
ax.set(xscale='log',xlabel='Efficiency multiplier x',ylabel='Duration of A (time units)',ylim=(0,106),title='(b) A conditional floor can move')
ax.legend(frameon=False,loc='upper right',fontsize=7.6)
for ax in axs:
 ax.set_xticks([1,10,100],labels=['1','10','100']);ax.grid(axis='y',alpha=.16)
out=P/'manuscript/figures';out.mkdir(exist_ok=True)
fig.savefig(out/'propagation_workflows.pdf')
plt.close(fig)
e=P/'evidence/propagation_revision';e.mkdir(exist_ok=True)
(e/'workflow_checks.json').write_text(json.dumps({'kind':'exact constructed illustration','baseline':[100,100],'at_x10':[19,91],'speedup_x10':[float(F(100,19)),float(F(100,91))],'elasticity_at_baseline':[0.9,0.1],'horizon_A':9,'workflow_B_horizon':'infinity on this path','empirical_claim':False},indent=2)+'\n')
print('Workflow checks passed; figure generated.')
