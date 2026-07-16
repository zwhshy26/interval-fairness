#!/usr/bin/env bash
set -euo pipefail

# Full experiment pipeline:
#   1. Optionally regenerate interval CSV files from temp/*.swf.gz
#   2. Run all experiments over temp/generated_intervals
#   3. Generate matplotlib figures and summary CSV
#
# Basic usage:
#   bash run_full_pipeline.sh
#
# Useful overrides:
#   RUNS=100 ALPHA=1 SEED=42 bash run_full_pipeline.sh
#   GENERATE=1 INTERVAL_NUM_SEEDS=5 bash run_full_pipeline.sh
#   GENERATE=1 INTERVAL_SEEDS=42,100,202 bash run_full_pipeline.sh
#   DATA_ROOT=temp/generated_intervals OUTPUT_ROOT=analysis_runs bash run_full_pipeline.sh

RUNS="${RUNS:-100}"
SEED="${SEED:-42}"
ALPHA="${ALPHA:-1}"
PYTHON="${PYTHON:-python3}"
GENERATE="${GENERATE:-0}"
INTERVAL_START_SEED="${INTERVAL_START_SEED:-42}"
INTERVAL_NUM_SEEDS="${INTERVAL_NUM_SEEDS:-1}"
INTERVAL_SEEDS="${INTERVAL_SEEDS:-}"
DATA_ROOT="${DATA_ROOT:-temp/generated_intervals}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis_runs}"
RUN_NAME="${RUN_NAME:-run_$(date +%Y%m%d_%H%M%S)}"

RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
DATA_DIR="${RUN_DIR}/data"
FIGURE_DIR="${RUN_DIR}/figures"
LOG_DIR="${RUN_DIR}/logs"
RESULT_CSV="${DATA_DIR}/experiment_results.csv"

mkdir -p "$DATA_DIR" "$FIGURE_DIR" "$LOG_DIR"

echo "Run directory: $RUN_DIR"
echo "Runs:          $RUNS"
echo "Seed:          $SEED"
echo "Alpha:         $ALPHA"
echo "Data root:     $DATA_ROOT"

if [[ "$GENERATE" == "1" ]]; then
  echo
  echo "Regenerating interval CSV files..."
  generate_args=(
    temp/File_reader.py
    --start-seed "$INTERVAL_START_SEED"
    --num-seeds "$INTERVAL_NUM_SEEDS"
  )

  if [[ -n "$INTERVAL_SEEDS" ]]; then
    generate_args+=(--seeds "$INTERVAL_SEEDS")
  fi

  "$PYTHON" "${generate_args[@]}" | tee "${LOG_DIR}/generate_intervals.log"
fi

echo
echo "Running experiments..."
"$PYTHON" main.py \
  --input "$DATA_ROOT" \
  --length-col length \
  --group-col group_id \
  --alpha "$ALPHA" \
  --runs "$RUNS" \
  --seed "$SEED" \
  --no-progress \
  --output "$RESULT_CSV" \
  | tee "${LOG_DIR}/experiments.log"

echo
echo "Generating matplotlib figures..."
"$PYTHON" plot_results.py \
  --input "$RESULT_CSV" \
  --output-dir "$FIGURE_DIR" \
  | tee "${LOG_DIR}/plots.log"

echo
echo "Done."
echo "Experiment CSV: $RESULT_CSV"
echo "Figures:        $FIGURE_DIR"
echo "Logs:           $LOG_DIR"
