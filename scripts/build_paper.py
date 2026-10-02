"""Reproduce local analyses and PDF without network or model calls."""
from pathlib import Path
import subprocess,sys,shutil,runpy,os,json,datetime
os.environ.setdefault("SOURCE_DATE_EPOCH", "1789826400")
P=Path(__file__).resolve().parents[1]
for script in ['analyze_review_evidence.py','analyze_joint_holdouts.py','check_joint_scores.py','analyze_identification_and_cost.py','analyze_historical_crossings.py','check_review_evidence.py','render_review_tables.py','render_joint_tables.py','analyze_capability_horizons.py','analyze_decision_pilot.py','analyze_pilot_threshold_sensitivity.py','check_validated_yield.py','check_theory_examples.py','check_examples.py','check_biological_example.py','plot_propagation.py','check_formal_transfer.py','check_agent_prices.py','plot_complexity_scenarios.py','analyze_descriptor_robustness.py','analyze_decision_robustness.py','analyze_finite_table_forecasts.py','check_eleven_review.py','render_eleven_review.py','analyze_endpoint_sensitivity.py','check_endpoint_sensitivity.py','render_gap_repairs.py','reconstruct_main_evidence.py','analyze_horizon_specification.py','verify_horizon_specification.py','render_horizon_specification.py']:
 runpy.run_path(str(P/'scripts'/script),run_name='__main__')
subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir=build','paper.tex'],cwd=P/'manuscript',check=True)
shutil.copy2(P/'manuscript/build/paper.pdf',P/'manuscript/paper.pdf')
runpy.run_path(str(P/'scripts/record_evidence.py'),run_name='__main__')
print('Reproduced paper PDF:',P/'manuscript/paper.pdf')
