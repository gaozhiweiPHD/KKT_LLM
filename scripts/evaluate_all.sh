#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results report
bash scripts/run.sh scripts/record_environment.py
bash scripts/run.sh -m pytest -q > results/pytest.log 2>&1
bash scripts/run.sh scripts/evaluate.py --split validation --limit 8 --methods qwen_sft --adapter checkpoints/sft/final --batch-size 1 --seeds 987 --output results/sft_fixed_validation > results/sft_fixed_validation.log 2>&1
for split in test_id test_ood; do
  bash scripts/run.sh scripts/evaluate.py --split "$split" --methods random greedy classical osqp --seeds 11 22 33 --output results/baselines > "results/baselines_$split.log" 2>&1
done
for method in zero sft grpo_A grpo_B grpo_C; do
  for split in test_id test_ood; do
    if [ "$method" = zero ]; then
      bash scripts/run.sh scripts/evaluate.py --split "$split" --methods "qwen_$method" --seeds 11 22 33 --batch-size 64 --output "results/qwen_$method" > "results/eval_""$method""_""$split"".log" 2>&1
    else
      bash scripts/run.sh scripts/evaluate.py --split "$split" --methods "qwen_$method" --adapter "checkpoints/$method/final" --seeds 11 22 33 --batch-size 64 --output "results/qwen_$method" > "results/eval_""$method""_""$split"".log" 2>&1
    fi
  done
done
bash scripts/run.sh scripts/verify_results.py > results/certificate_audit.log 2>&1
bash scripts/run.sh scripts/plot_results.py > results/plots.log 2>&1
bash scripts/run.sh scripts/analyze_failures.py > results/case_studies.log 2>&1
bash scripts/run.sh scripts/build_report.py > results/report.log 2>&1
