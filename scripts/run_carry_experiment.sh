#!/usr/bin/env bash
# Run from the repository: bash scripts/run_carry_experiment.sh ppo 0,1,4,5 8192 4000 ppo_baseline
set -euo pipefail
if (( $# < 5 )); then
  echo 'Usage: bash scripts/run_carry_experiment.sh {ppo|supermimic} gpu_csv envs iterations run_name [train flags]' >&2
  exit 2
fi
learner=$1; gpu_list=$2; envs=$3; iterations=$4; run_name=$5
shift 5
[[ $learner == ppo || $learner == supermimic ]] || { echo 'Unknown learner' >&2; exit 2; }
[[ $gpu_list =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Use physical GPU numbers separated by commas' >&2; exit 2; }
[[ $envs =~ ^[1-9][0-9]*$ && $iterations =~ ^[1-9][0-9]*$ ]] || exit 2
[[ $run_name =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Use a short alphanumeric run name' >&2; exit 2; }

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n "${CARRY_ENV_SH:-}" ]]; then
  # Optional site-specific CUDA/mjlab exports used on the original host.
  source "$CARRY_ENV_SH"
fi
cd "$repo"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=disable WANDB_MODE=disabled OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="$gpu_list"
export TMPDIR="${TMPDIR:-$repo/.cache/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$repo/.cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$repo/.cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$repo/.cache/triton}"
out="${CARRY_OUTPUT_DIR:-$repo/runs}"
mkdir -p "$out" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
prefix="$out/$run_name"
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${prefix}_start_utc.txt"
(
  while :; do
    printf '%s,' "$(date -u +%s)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | paste -sd ';' -
    sleep 2
  done
) > "${prefix}_gpu_samples.csv" 2> "${prefix}_gpu_monitor.log" &
monitor=$!
trap 'kill "$monitor" 2>/dev/null || true' EXIT
IFS=, read -ra devices <<< "$gpu_list"
if (( envs % ${#devices[@]} != 0 )); then
  echo 'Global environment count must be divisible by GPU count' >&2
  exit 2
fi
python_bin="${CARRY_PYTHON:-python}"
if (( ${#devices[@]} > 1 )); then
  port="${CARRY_MASTER_PORT:-29571}"
  launch=("$python_bin" -m torch.distributed.run --nnodes=1 --nproc_per_node="${#devices[@]}" --master_addr=127.0.0.1 --master_port="$port")
else
  launch=("$python_bin")
fi
compile=--no-compile
[[ $learner == supermimic ]] && compile=--compile
set +e
"${launch[@]}" src/mjlab_g1/scripts/train_track.py G1-SMTrack-CarryPlace \
  --learner "$learner" --num-envs "$envs" --max-iterations "$iterations" \
  --seed 0 --start-uniform-mix 0.5 --save-interval "${CARRY_SAVE_INTERVAL:-100}" \
  --run-name "$run_name" "$compile" "$@" > "${prefix}.log" 2>&1
code=$?
set -e
date -u +'%Y-%m-%dT%H:%M:%SZ' > "${prefix}_end_utc.txt"
echo "$code" > "${prefix}_exit_code.txt"
tail -35 "${prefix}.log"
exit "$code"
