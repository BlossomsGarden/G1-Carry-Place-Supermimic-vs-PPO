#!/usr/bin/env bash
# Evaluate the shipped model and save both sparse outcome and handling diagnostics.
set -euo pipefail
if (( $# < 1 || $# > 5 )); then
  echo 'Usage: bash scripts/eval_checkpoints.sh {ppo|supermimic} [physical_gpu=0] [seed=2027] [episodes=64] [optional_checkpoint]' >&2
  exit 2
fi
learner=$1; gpu=${2:-0}; seed=${3:-2027}; episodes=${4:-64}
[[ $gpu =~ ^[0-9]+$ && $seed =~ ^[0-9]+$ && $episodes =~ ^[1-9][0-9]*$ ]] || exit 2
case $learner in
  ppo) model=model_3999.pt ;;
  supermimic) model=model_1000.pt ;;
  *) echo 'learner must be ppo or supermimic' >&2; exit 2 ;;
esac
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ -n ${CARRY_ENV_SH:-} ]]; then source "$CARRY_ENV_SH"; fi
cd "$repo"
checkpoint=${5:-checkpoints/$learner/$model}
if [[ ! -f $checkpoint ]] || (( $(wc -c < "$checkpoint") < 1000000 )); then
  echo "Missing model bytes at $checkpoint; download the checkpoint from Hugging Face (see checkpoints/README.md)" >&2
  exit 1
fi
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=disable CUDA_VISIBLE_DEVICES="$gpu"
export TMPDIR="${CARRY_TMPDIR:-$repo/.cache/tmp}"
export XDG_CACHE_HOME="${CARRY_XDG_CACHE_HOME:-$repo/.cache/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${CARRY_INDUCTOR_CACHE_DIR:-$repo/.cache/torchinductor}"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR"
python_bin=${CARRY_PYTHON:-python}
prefix="runs/evaluations/${learner}_${seed}_n${episodes}"
mkdir -p runs/evaluations
"$python_bin" src/mjlab_g1/scripts/eval_track.py \
  --task G1-SMTrack-CarryPlace --learner "$learner" --checkpoint "$checkpoint" \
  --episodes "$episodes" --seed "$seed" --randomize \
  --rollout-all-npz "${prefix}.npz" --output "${prefix}.json"
"$python_bin" scripts/score_carry_object_outcome.py \
  --rollout "${prefix}.npz" --eval-json "${prefix}.json" \
  --motion dataset/tracking/carry_place_refs/carry_place_sugar_098_refined_v2.npz \
  --output "${prefix}_metrics.json"
echo "Physical evaluation: ${prefix}_metrics.json"
