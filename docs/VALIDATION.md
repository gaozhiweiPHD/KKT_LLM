# Release validation — 2026-11-16

Validation used the staged release code and archives, leaving the source experiment unchanged. Machine-readable results are in [validation_results.json](validation_results.json).

| Check | Result |
|---|---|
| Original regression suite before added release tests | 66 passed |
| Full suite including artifact safety and residual-audit tests | 77 passed |
| Fresh virtual environment, installed with `pip install -e '.[dev]'` | 72 CPU tests passed; no Torch/Transformers/TRL installed |
| Numerical smoke tests | Known-active, OSQP and greedy examples passed |
| Two-dimensional demo | ADD(1), ADD(0), STOP; x=[1,1]; all residuals zero |
| Archive restoration into empty working directories | All 159 files verified and restored |
| Independent certificate audit | 40,500 candidates; zero KKT/success mismatches |
| Training evidence | All 600 GRPO groups audited; matches reference |
| Restored GRPO-B adapter, pinned base model loaded offline | Two ID problems, seed 11, batch size 2; both successful |
| Packaging | Python syntax, TOML, shell syntax, local documentation links checked |

The full suite used the original Python 3.13.9 training environment. The fresh numerical environment installed NumPy 2.5.5, SciPy 1.18.1, OSQP 1.1.3 and pytest 9.2.0. Both produced 12 existing SciPy warnings about forwarding the `threads` option to HiGHS.

The first historical audit attempt exposed a portability issue: a fixed 1e-12 absolute comparison of raw residuals was too strict under cancellation with large multipliers on a different CPU/BLAS kernel. For example, a failed random-policy candidate with a multiplier around 350,000 had saved/recomputed raw stationarity residuals about 9.54e-11/8.08e-11. The release audit now allows differences bounded by float64 arithmetic error scaled to the absolute operands and logs every reconciliation.

There were **92 raw-residual reconciliations**; KKT acceptance still uses the original 1e-5 normalized threshold. All original success labels and counts at the three audited tolerances (1e-5, 1e-7, 1e-9) match exactly, as do all saved training-audit summaries. The full evidence is in [release_certificate_audit.json](release_certificate_audit.json). No baseline or learned-policy success was changed by this packaging fix.

The GPU check validates archive restoration, cache selection, model/adapter loading and interactive inference. Two problems are not a new performance estimate. Training and the full model evaluation were not rerun. The original result tables remain under `reference/`.
