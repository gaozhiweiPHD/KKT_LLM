# Release artifacts and provenance

The Git repository contains code and a compact frozen evidence tree. Larger reproducibility files are attached to GitHub Release `v0.1.0`. The authoritative names, sizes and SHA-256 hashes are in [`release_manifest.json`](release_manifest.json).

| Archive | Extracts to | Contents |
|---|---|---|
| `kkt-data-v0.1.0.tar.gz` | `data/` | 7,000 QPs, separate oracle/metadata/audit files, teacher trajectories, SFT rows, curriculum and manifest |
| `kkt-checkpoints-v0.1.0.tar.gz` | `checkpoints/` | SFT and GRPO-A/B/C final adapters, tokenizer assets; GRPO optimizer/scaler/RNG states |
| `kkt-evidence-v0.1.0.tar.gz` | `results/`, `checkpoints/` | Original formal per-episode CSVs, full action trajectories, setup files, aggregate audits/figures, teacher summaries, training/validation/rollout logs |

The checkpoint archive does not contain base Qwen weights, repeated intermediate checkpoints, or the SFT `training_args.bin` pickle (not required for inference). Base weights are fetched at the revision in `configs/model_revision.txt`. Original adapter metadata is preserved; `load_model` independently enforces the recorded base revision even where the original SFT adapter config omitted it.

The evidence archive includes the historical invalidated `sft_validation` pilot and its `INVALIDATED.json` marker because the original report documents that mistake. Formal results use `sft_fixed_validation` and the correctly labelled test evaluations. No invalidated pilot enters the main table.

Use `scripts/restore_artifacts.py` to verify and extract. It does not download files or load model pickle state. It accepts regular files only, rejects paths outside the intended top-level directories, checks hashes before writing, and rejects differing existing files by default. Restore in a clean checkout to keep saved and newly generated runs distinct.

[`reference/source_manifest.json`](../reference/source_manifest.json) records source-file hashes before release-specific code/documentation changes. [`PACKAGING.md`](PACKAGING.md) explains those changes. `reference/` results are historical observations, not newly produced performance measurements.

The original experiment files and local caches were left intact during packaging. Virtual environments, package/model caches, credentials, installation logs, temporary LaTeX builds and repeated checkpoints are not part of the release.
