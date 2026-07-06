#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_experiments.sh
#
# Optional environment variables:
#   RUNS=10 SEED=42 ALPHA=1.5 DATA_ROOT=temp/generated_intervals bash run_experiments.sh

RUNS="${RUNS:-10}"
SEED="${SEED:-42}"
ALPHA="${ALPHA:-1.0}"
PYTHON="${PYTHON:-python3}"
DATA_ROOT="${DATA_ROOT:-temp/generated_intervals}"

mapfile -t DATASETS < <(find "$DATA_ROOT" -type f -name "*.csv" | sort)

mkdir -p results

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No CSV files found under $DATA_ROOT"
  exit 1
fi

for dataset in "${DATASETS[@]}"; do
  relative_name="${dataset#"$DATA_ROOT"/}"
  name="${relative_name%.csv}"
  name="${name//\//__}"
  output="results/${name}_runs${RUNS}_alpha${ALPHA}.txt"
  result_csv="results/${name}_runs${RUNS}_alpha${ALPHA}.csv"

  args=(
    main.py
    --input "$dataset"
    --length-col length
    --group-col group_id
    --runs "$RUNS"
    --seed "$SEED"
    --alpha "$ALPHA"
    --output "$result_csv"
    --no-progress
  )

  echo "Running $dataset"
  "$PYTHON" "${args[@]}" | tee "$output"
  echo "Saved $output"
  echo "Saved $result_csv"
done
