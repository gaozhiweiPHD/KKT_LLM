# Frozen reference experiment

These are saved observations from the original Qwen3/KKT experiment. The release packaging did not rerun training or replace these measurements.

- [Research report (Chinese)](report/research_report.md)
- [Success and failure cases](report/case_studies.md)
- [Lecture slides](report/kkt_guided_qwen3_lecture.pdf)
- [Matched per-episode records](results/matched_evaluations.csv)
- [Comparison across inference seeds](results/comparison.csv)
- [Dataset construction audit](data/manifest.json)
- [Certificate/rollout audit](results/certificate_audit.json)
- [Original package/GPU versions](results/environment.json)
- [Training diagnostics](results/training_diagnostics.csv)

`checkpoints/` here contains small logs and parameter audits, not weights. `results/` contains summaries and plots; full raw trajectories are in the evidence archive. The root-level working directories are populated by artifact restore or a new run. See [reproduction](../docs/REPRODUCING.md), [mathematical method](../docs/METHOD.md), and [release validation](../docs/VALIDATION.md).

The original report includes historical pilot corrections and environment details. The core result is one training seed, not three independent training runs per method. Its three seeds refer to inference sampling.
