# Release packaging changes

The release was prepared from the existing `kkt_qwen_agent` experiment. The source directory, model caches and original checkpoints were preserved.

Changes made for reproduction:

- Separate English/Chinese introductions, an English method explanation, the original detailed Chinese mathematical discussion, and explicit reproduction/artifact guides.
- Portable Python and cache selection in `scripts/run.sh` and `kkt_agent/llm.py`, respecting caller-provided cache locations. The default Qwen revision remains pinned even outside an editable checkout.
- Declared numerical package dependencies and optional training/analysis/development extras.
- A sequential `scripts/reproduce.sh` entry point, a two-dimensional numerical demo, environment capture, and CPU CI configuration.
- Evaluation creates its output directories before log redirection. Newly generated reports use the saved current environment instead of copying historical cluster-specific narration.
- Historical residual comparisons now allow float64 arithmetic roundoff scaled by the absolute dot-product operands. Every such reconciliation is logged; KKT thresholds, normalized-residual comparisons and success labels are unchanged. This addresses cancellation in large-multiplier failure cases across CPU/BLAS kernels.
- Frozen reference evidence is separate from writable runtime outputs; large data/checkpoint/trajectory files are checksum-verified Release attachments.

The QP generator, active-set transitions, numerical solves, verifier equations, rewards, SFT objective and main GRPO objective were not changed. Paths and packaging changes can be reviewed against the original-file hashes in `reference/source_manifest.json`. Checkpoint bytes and archived data/evidence bytes are preserved.

`requirements.lock.txt` remains the historical exact environment snapshot. CPU CI uses the declared lightweight dependencies; it is a portability check, not a repetition of the GPU training experiment. No new project-wide license has been selected as part of this packaging; upstream model metadata and notices are preserved.
