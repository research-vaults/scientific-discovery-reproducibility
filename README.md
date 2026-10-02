# Position: Toward a Computational Theory of Scientific Discovery

Which scientific problems become solvable as AI improves, and when do validated results follow? This paper connects capability growth to target success, procedure availability and validated completion. Its retrospective studies show why useful forecasting measurements depend on the target and evaluation budget, including cases where additional information worsens predictions.

**[Read the paper (PDF)](manuscript/paper.pdf)** · [LaTeX source](manuscript/paper.tex) · Artifact: **2 October 2026**

The PDF matches the **30 September 2026 development source** reproduced here. It is a later research draft, not the accepted AgenticLS submission attachment. Pages 1–9 contain the main paper and disclosure, 10–13 references, and 14–30 the appendix. The PDF's build timestamp is fixed for reproducibility; it is not its revision date.

## Setup and reproduction

Use Python 3.11 and TeX Live with `latexmk`. Run in a disposable checkout because reproduction overwrites generated results, tables and figures:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/fetch_sources.py
python scripts/build_paper.py
```

Expected output: numerical results under `evidence/`, plots and tables under `manuscript/figures/`, and the 30-page `manuscript/paper.pdf`. The complete fixed replay took a few minutes on the release machine. It uses CPU computation, no GPU, API keys, paid calls or wet-lab experiments. Allow about 200 MB for the checkout, downloaded inputs and generated outputs, excluding Python/TeX installations; peak RAM was not measured. Internet access is needed for installation and checksum-verified source retrieval, then the analyses run locally.

To rebuild only the paper from the supplied figures and tables:

```sh
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build paper.tex
```

This writes `manuscript/build/paper.pdf`. The full pipeline additionally copies it to the linked `manuscript/paper.pdf`.

## Analysis map

All commands below run in the full pipeline after source retrieval.

| Question or check | Script |
|---|---|
| Do target descriptors improve forecasts across models and reasoning families? | `scripts/analyze_review_evidence.py`, `scripts/analyze_joint_holdouts.py` |
| Do early outcomes improve forecasts in fixed chemical tables? | `scripts/analyze_finite_table_forecasts.py`, `scripts/analyze_endpoint_sensitivity.py` |
| How do validation cost and uncertainty affect procedure comparisons? | `scripts/analyze_identification_and_cost.py`, `scripts/analyze_pilot_threshold_sensitivity.py` |
| Which conclusions depend on the horizon specification? | `scripts/analyze_horizon_specification.py`, `scripts/verify_horizon_specification.py` |
| Can the reported calculations be reconstructed independently? | `scripts/check_joint_scores.py`, `scripts/check_eleven_review.py`, `scripts/reconstruct_main_evidence.py` |

Reference outputs are retained in `evidence/`; historical directory names remain to preserve working script paths. The source contracts record exploratory status and analysis assumptions. [Validation](VALIDATION.md) describes the completed clean-checkout replay and its limits; [SHA256SUMS](SHA256SUMS) records the distributed files.

## Inputs and interpretation

- **ObsScaling:** aggregate evaluation scores from [commit `4d6e1e43fd26`](https://github.com/ryoungj/ObsScaling/tree/4d6e1e43fd2635d04654aa77d1df9d5266ea0382), with its Apache-2.0 licence and attribution. Only the required subset is bundled; notebook links lead upstream.
- **Chemistry:** [SOURCE_DATA.json](SOURCE_DATA.json) pins [GOLLuM](https://github.com/schwallergroup/gollum/tree/c418d7ed3c17e5995f503f6df7c26e9f7c58d09f) inputs. Raw chemistry tables and detailed outcome traces are not distributed; retrieval and replay reconstruct them locally. Consult upstream dataset terms—a code licence alone does not establish all data rights.
- **METR:** [METR_INPUTS.json](METR_INPUTS.json) pins the historical diagnostic inputs. They are retrieved rather than redistributed because an explicit redistribution licence was not established.

These are fixed retrospective analyses and conditional constructions, not independent replication or prospective validation of scientific completion dates. They do not establish a universal complexity scale or clinical forecasts. If a source checksum fails, obtain the recorded version rather than substituting new data.

## Citation and rights

For this artifact, cite: *Position: Toward a Computational Theory of Scientific Discovery*. Development revision 30 September 2026; reproducibility artifact 2 October 2026. [Repository](https://github.com/research-vaults/scientific-discovery-reproducibility). Include the commit used when reporting reproduced results.

Original code is MIT-licensed under [LICENSE](LICENSE); manuscript prose and original figures are CC BY 4.0. Third-party code, data and bibliographic material retain their own terms and attribution. The project licence does not relicense upstream observations. The public artifact retains the source manuscript's anonymous author field; it is not a blind-review package.
