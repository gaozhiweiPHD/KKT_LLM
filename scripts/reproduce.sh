#!/usr/bin/env bash
# Use a GPU allocation for training and model evaluation.
set -euo pipefail
cd "$(dirname "$0")/.."
stage="${1:-help}"
run() { bash scripts/run.sh "$@"; }
case "$stage" in
  cpu)
    run -m pytest -q --ignore=tests/test_grpo_gradient.py --ignore=tests/test_grpo_math.py
    run scripts/smoke_test.py
    ;;
  data)
    run scripts/generate_dataset.py
    run scripts/calibrate_tolerance.py
    run scripts/generate_teacher.py
    run scripts/generate_teacher.py --split validation
    run scripts/build_curriculum.py
    ;;
  train)
    run scripts/record_environment.py
    run scripts/check_api.py
    run scripts/train_sft.py --steps 200 --output checkpoints/sft
    run scripts/evaluate_actions.py --adapter checkpoints/sft/final --output results/sft_action_accuracy.json
    for reward in A B C; do
      run scripts/train_grpo.py --steps 200 --reward "$reward"
    done
    ;;
  evaluate)
    bash scripts/evaluate_all.sh
    ;;
  audit)
    mkdir -p report
    run scripts/verify_results.py
    run scripts/plot_results.py
    run scripts/analyze_failures.py
    run scripts/build_report.py
    ;;
  all)
    for part in data train evaluate; do bash scripts/reproduce.sh "$part"; done
    ;;
  help|-h|--help)
    echo "Usage: bash scripts/reproduce.sh {cpu|data|train|evaluate|audit|all}"
    echo "See docs/REPRODUCING.md for dependencies, artifacts and GPU requirements."
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
