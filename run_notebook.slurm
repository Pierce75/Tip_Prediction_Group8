#!/bin/bash
#SBATCH --job-name=tip_notebook
#SBATCH --partition=cs4207
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/notebook_%j.out
#SBATCH --error=logs/notebook_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Use the project environment created in the README, or a supplied interpreter.
python_bin="${TIP_PYTHON:-$PWD/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
    echo "Set up .venv or export TIP_PYTHON with the path to your Python executable." >&2
    exit 1
fi
export PATH="$(dirname "$python_bin"):$PATH"

shared_data=/home/cs4207/data/yellow_tripdata_2026-01.parquet
if [[ -z "${TIP_DATA_FILE:-}" && -f "$shared_data" ]]; then
    export TIP_DATA_FILE="$shared_data"
fi

run_dir="$PWD/runs/$SLURM_JOB_ID"
mkdir -p "$run_dir"
export TIP_OUTPUT_DIR="$run_dir/outputs"
work_dir="$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/cs4207-$SLURM_JOB_ID.XXXXXXXX")"
trap 'rm -rf -- "$work_dir"' EXIT
export TIP_WORK_DIR="$work_dir"

"$python_bin" -m nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=1800 --output notebook --output-dir "$run_dir" notebook.ipynb
