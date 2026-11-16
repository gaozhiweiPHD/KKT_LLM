# Reproducing the experiment

Run commands from the repository root. All stages are sequential and use one CPU thread per numerical library. On a shared cluster, run training and large evaluations inside an allocated GPU job. The scripts do not submit jobs automatically.

## 1. Choose an environment

The recorded environment was Linux, Python **3.13.9**, PyTorch 2.8.0 / CUDA 12.8, Transformers 5.20.2, TRL 1.16.0 and PEFT 0.21.2, with one Quadro RTX 6000. Use a CUDA GPU for the FP16 training scripts. CPU-only numerical checks need no model or CUDA. Other Python/package/GPU combinations may work but are not the recorded environment.

For the full experiment:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
bash scripts/run.sh -m pytest -q
```

`requirements.lock.txt` is the exact original Linux/CUDA package snapshot. If a pinned build is unavailable from your package index, install `requirements.txt` or `pip install -e '.[train,analysis,dev]'` to create a compatible environment, record its versions and treat it as a new environment. Do not silently claim that such an installation reproduces the original dependency snapshot.

For numerical checks only (Python 3.11+):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
bash scripts/reproduce.sh cpu
bash scripts/run.sh examples/active_set_demo.py
```

`run.sh` chooses `KKT_PYTHON` if set, otherwise an active virtual environment, then the repository's `.venv`, then `python3`. It sets local cache defaults under `.cache/` while honoring your existing `HF_HOME`, `HF_HUB_CACHE`, `TORCH_HOME`, and `MPLCONFIGDIR`. It resolves paths relative to the repository, so it also works when invoked from elsewhere.

## 2. Download the original artifacts (optional)

The [v0.1.0 release](https://github.com/gaozhiweiPHD/KKT_LLM/releases/tag/v0.1.0) has three archives. Download with GitHub CLI (authenticate first if the repository is private):

```bash
mkdir -p downloads
gh release download v0.1.0 --repo gaozhiweiPHD/KKT_LLM \
  --pattern 'kkt-*.tar.gz' --dir downloads
python scripts/restore_artifacts.py --directory downloads data checkpoints evidence
```

The restore script checks each archive's SHA-256 against the manifest committed in this repository, checks member hashes, and refuses to overwrite differing files unless `--overwrite` is explicitly supplied. For an offline audit, download/restore just `data` and `evidence`; for model evaluation use `data` and `checkpoints`. All three together give the original working layout.

Base Qwen weights are not in the archives. `load_model` downloads `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`. Internet is needed for the first model load; an already populated Hugging Face cache can be reused offline. Do not substitute a moving model revision for a matched reproduction.

## Route A: inspect and audit the recorded run

The repository's `reference/` already contains the main report, per-episode matched CSV, training metrics and plots. No model download is needed to read these.

To recompute certificates and regenerate tables/plots from all saved trajectories, restore `data` and `evidence`, then install the numerical/analysis dependencies:

```bash
python -m pip install -e '.[analysis,dev]'
bash scripts/reproduce.sh audit
```

This recomputes certificates for **40,500** evaluated episodes and audits the 600 GRPO update groups (four rollouts each). It uses saved model actions, not new inference. The original fixed raw-residual comparison can fail across CPU/BLAS kernels due to cancellation; the release audit logs arithmetic-scale roundoff reconciliations while requiring unchanged KKT/STOP success decisions. The evidence archive provides the original environment and historical test log, so regenerated reports continue to describe that saved experiment. Fresh test results are recorded separately in `docs/VALIDATION.md`.

## Route B: evaluate the trained adapters

Use the full environment, restore `data` and `checkpoints`, and run a small pilot on a GPU:

```bash
bash scripts/run.sh scripts/evaluate.py \
  --split test_id --limit 8 --methods qwen_grpo_B \
  --adapter checkpoints/grpo_B/final --seeds 11 --batch-size 8 \
  --output results/pilot_grpo_B
```

The pilot is a functional check, not the full result table. For all reference methods and both full test splits:

```bash
bash scripts/reproduce.sh evaluate
```

This runs numerical baselines and Qwen zero-shot/SFT/GRPO-A/B/C with inference seeds 11/22/33, batch size 64 for the main LLM runs, and 20 decisions per episode. It verifies final candidates, creates matched tables and figures, and writes `report/research_report.md`. Failed agents never fall back to OSQP.

If memory is insufficient, use a smaller `--batch-size` in individual `evaluate.py` commands (or edit `evaluate_all.sh`). Changing generation batch shape can change FP16 logits and sampled trajectories; record the change and compare statistics, not bitwise output. Original per-method setup files in `reference/results/` record peak allocated memory, batch sizes and timings.

## Route C: regenerate data and retrain

Use a fresh checkout/working directory, or keep downloaded original artifacts separate. Training overwrites the corresponding output paths; the frozen `reference/` files remain separate.

```bash
bash scripts/reproduce.sh data
bash scripts/reproduce.sh train
bash scripts/reproduce.sh evaluate
```

Or run all three with `bash scripts/reproduce.sh all`.

The data stage generates 5,000 train / 500 validation / 1,000 ID / 500 OOD QPs; checks each with OSQP; calibrates tolerance; creates teacher demonstrations; and prepares curriculum initial states. Seeds are disjoint across splits. Oracle answers are kept in separate files and are only read by offline preparation.

The training stage runs SFT for 200 updates and independently starts three 200-update GRPO runs from `checkpoints/sft/final`. The seed is 42, LoRA rank/alpha are 8/16, learning rates are 1e-4 (SFT) and 1e-5 (GRPO), group size is 4, and the main action dialect is plain text. Config files and `parameter_audit.json` record settings. The optional tool-calling TRL backend is not part of this route.

Original recorded loop times were about 384.5 s for SFT and 1,736–1,760 s for each GRPO variant on the reference GPU, including periodic validation and saving but excluding model loading/preprocessing. These are observations, not a runtime guarantee. Full dataset preparation and evaluation add time. Repeated training on other hardware need not reproduce identical parameters or actions.

## Short training and resume checks

After the data stage, short checks can be written to separate directories:

```bash
bash scripts/run.sh scripts/train_sft.py --steps 4 --output checkpoints/sft_smoke
bash scripts/run.sh scripts/train_grpo.py --sft checkpoints/sft_smoke/final \
  --steps 2 --reward B --curriculum-level 0 --eval-every 0 --save-every 2 \
  --output checkpoints/grpo_smoke
bash scripts/run.sh scripts/train_grpo.py \
  --resume checkpoints/grpo_smoke/final --steps 4 --reward B \
  --curriculum-level 0 --eval-every 0 --output checkpoints/grpo_smoke
```

`--steps` is the total target number of updates, not additional updates. GRPO checkpoints save optimizer, scaler and random states. The released SFT `final` directory is for adapter loading; resuming SFT requires a Trainer `checkpoint-*` directory created by your own run. Original intermediate checkpoints are omitted from the release.

## Check what was reproduced

- **Numerical correctness:** unit tests, generated-data OSQP audit and four independent KKT residuals.
- **Learning:** nonzero gradients/parameter changes, logged group variance and held-out success. A near-zero single-update GRPO scalar loss is expected and is not alone evidence of no learning.
- **Comparability:** match `(problem_id, inference seed)`, model revision, batch size, action budget and tolerance. Use per-seed statistics and stratified metrics.
- **Scope:** the original result uses one training seed and three inference seeds. A faithful rerun is not a multi-training-seed validation.

`results/` and `report/` are writable outputs; `reference/` is the frozen comparison. See [ARTIFACTS.md](ARTIFACTS.md) and [PACKAGING.md](PACKAGING.md) for provenance and release-specific changes.
