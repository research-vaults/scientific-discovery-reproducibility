"""Render the complete main diagnostic grid from saved, verified scores."""
from pathlib import Path
import csv
P=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader((P/'evidence/three_review_remediation_2026-09-26_221254/results/summary.csv').open()))
ix={(float(r['quantile']),int(r['pilot']),int(r['budget']),r['predictor'],r['stratum']):r for r in rows}
lines=[r'\begin{tabular}{lrrrrrrrr}',r'\toprule',r'Endpoint & Pilot & Budget & \multicolumn{3}{c}{Budget-only} & \multicolumn{3}{c}{Summaries}\\',r' & & & Shared & Separate & Matched & Shared & Separate & Matched\\',r'\midrule']
for q in [.9,.95,.99]:
 for pilot in [5,10]:
  for budget in [20,40]:
   vals=[float(ix[q,pilot,budget,name+'__'+v,'all']['brier']) for name in ['no_target','pilot_summary'] for v in ['pooled','separate','separate_matched_penalty']]
   lines.append(f'{int(q*100)} & {pilot} & {budget} & '+' & '.join(f'{v:.5f}' for v in vals)+r'\\')
lines += [r'\bottomrule',r'\end{tabular}']
(P/'manuscript/figures/horizon_specification.tex').write_text('\n'.join(lines)+'\n')
