#!/usr/bin/env bash
# Run the ViFi-CLIP external baselines on WSL.
#
# Default behavior:
# - reads model ids and seeds from external_baselines/experiments/baselines.yaml
# - selects ViFi-CLIP models whose frames field equals VIFI_FRAMES, default 8
# - runs train, then val, then test for each model/seed
# - writes batch-level logs under external_baselines/run_logs/batch/
# - run_experiment.py writes stable stage logs under external_baselines/run_logs/<model>/seed<seed>/
# - stops on the first failure, prints the failing command, and pauses if interactive
#
# Useful overrides:
# - VIFI_FRAMES=16 runs only vifi_clip_16f
# - VIFI_FRAMES=all runs every ViFi-CLIP entry registered in baselines.yaml
# - EVAL_NUM_CROP=1 switches standalone val/test to center-crop single-view evaluation
# - CLASS_BALANCED_LOSS=1 enables class-balanced CE for every stage log/config
# - CLASS_BALANCED_LOSS=0 disables class-balanced CE even if baselines.yaml enables it

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXTERNAL_ROOT}/.." && pwd)"
BASELINE_FILE="${EXTERNAL_ROOT}/experiments/baselines.yaml"

CONDA_ENV="${CONDA_ENV:-retail_vificlip}"
CONFIG_PYTHON="${CONFIG_PYTHON:-python}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
PAUSE_ON_EXIT="${PAUSE_ON_EXIT:-1}"
EVAL_NUM_CROP="${EVAL_NUM_CROP:-3}"
VIFI_FRAMES="${VIFI_FRAMES:-8}"
CLASS_BALANCED_LOSS="${CLASS_BALANCED_LOSS:-auto}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${VIFI_FRAMES}" == "all" || "${VIFI_FRAMES}" == "*" || -z "${VIFI_FRAMES}" ]]; then
  LOG_FRAME_TAG="all"
else
  LOG_FRAME_TAG="${VIFI_FRAMES%f}f"
fi
LOG_DIR="${EXTERNAL_ROOT}/run_logs/batch/vificlip_${LOG_FRAME_TAG}_3seed_${RUN_ID}"

pause_if_needed() {
  if [[ "${PAUSE_ON_EXIT}" == "1" && -t 0 ]]; then
    read -r -p "Press Enter to close this WSL session..." _
  fi
}

on_interrupt() {
  echo
  echo "[interrupt] Run interrupted. Logs so far: ${LOG_DIR}"
  pause_if_needed
  exit 130
}

trap on_interrupt INT TERM

run_and_log() {
  local name="$1"
  shift
  local log_file="${LOG_DIR}/${name}.log"

  echo
  echo "===== START ${name} ====="
  echo "[log] ${log_file}"
  echo "[cmd] $*"

  "$@" 2>&1 | tee "${log_file}"
  local status="${PIPESTATUS[0]}"

  if [[ "${status}" -eq 0 ]]; then
    echo "===== DONE ${name} ====="
  else
    echo "===== FAILED ${name} exit=${status} ====="
    echo "[failed log] ${log_file}"
  fi
  return "${status}"
}

build_class_balanced_args() {
  CLASS_BALANCED_ARGS=()
  case "${CLASS_BALANCED_LOSS,,}" in
    ""|"auto")
      ;;
    "1"|"true"|"yes"|"on")
      CLASS_BALANCED_ARGS+=(--class-balanced-loss)
      ;;
    "0"|"false"|"no"|"off")
      CLASS_BALANCED_ARGS+=(--no-class-balanced-loss)
      ;;
    *)
      echo "[error] Invalid CLASS_BALANCED_LOSS=${CLASS_BALANCED_LOSS}. Use auto, 1, or 0."
      pause_if_needed
      return 2
      ;;
  esac
}

main() {
  if [[ "${EVAL_NUM_CROP}" != "1" && "${EVAL_NUM_CROP}" != "3" ]]; then
    echo "[error] EVAL_NUM_CROP must be 1 or 3, got ${EVAL_NUM_CROP}."
    pause_if_needed
    return 2
  fi

  build_class_balanced_args || return $?

  mkdir -p "${LOG_DIR}"
  cd "${PROJECT_ROOT}"

  mapfile -t MODELS < <("${CONFIG_PYTHON}" -c "import json, sys
cfg = json.load(open(sys.argv[1], encoding='utf-8-sig'))
frame_filter = sys.argv[2].strip().lower()
wanted = None
if frame_filter not in ('', 'all', '*'):
    value = frame_filter[:-1] if frame_filter.endswith('f') else frame_filter
    try:
        wanted = int(value)
    except ValueError:
        raise SystemExit(f'Invalid VIFI_FRAMES={sys.argv[2]!r}; use 8, 16, or all.')
for model in cfg['models']:
    if model.get('backend') != 'vifi_clip':
        continue
    if wanted is not None and int(model.get('frames', -1)) != wanted:
        continue
    print(model['model_id'])" "${BASELINE_FILE}" "${VIFI_FRAMES}")
  mapfile -t SEEDS < <("${CONFIG_PYTHON}" -c "import json, sys; cfg=json.load(open(sys.argv[1], encoding='utf-8-sig')); print('\n'.join(str(s) for s in cfg['defaults'].get('seeds', [])))" "${BASELINE_FILE}")

  if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "[error] No ViFi-CLIP models found in ${BASELINE_FILE} for VIFI_FRAMES=${VIFI_FRAMES}."
    pause_if_needed
    return 2
  fi
  if [[ "${#SEEDS[@]}" -eq 0 ]]; then
    echo "[error] No seeds found in ${BASELINE_FILE} defaults.seeds."
    pause_if_needed
    return 2
  fi

  echo "[project] ${PROJECT_ROOT}"
  echo "[baseline] ${BASELINE_FILE}"
  echo "[logs] ${LOG_DIR}"
  echo "[conda env] ${CONDA_ENV}"
  echo "[models] ${MODELS[*]}"
  echo "[seeds] ${SEEDS[*]}"
  echo "[vifi frames] ${VIFI_FRAMES}"
  echo "[eval num crop] ${EVAL_NUM_CROP}"
  echo "[class balanced loss] ${CLASS_BALANCED_LOSS}"
  echo "[stop on error] ${STOP_ON_ERROR}"

  local failures=0
  local status=0
  local model=""
  local seed=""
  local stage=""
  local cmd=()

  for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      for stage in train val test; do
        cmd=(
          conda run --no-capture-output -n "${CONDA_ENV}" python \
          external_baselines/scripts/run_experiment.py \
          --model "${model}" \
          --seed "${seed}" \
          --stage "${stage}"
        )
        if [[ "${stage}" != "train" ]]; then
          cmd+=(--num-crop "${EVAL_NUM_CROP}")
        fi
        if [[ "${#CLASS_BALANCED_ARGS[@]}" -gt 0 ]]; then
          cmd+=("${CLASS_BALANCED_ARGS[@]}")
        fi

        run_and_log "${model}_seed${seed}_${stage}" "${cmd[@]}"
        status=$?
        if [[ "${status}" -ne 0 ]]; then
          failures=$((failures + 1))
          if [[ "${STOP_ON_ERROR}" == "1" ]]; then
            echo
            echo "[stop] First failure encountered at ${model} seed${seed} ${stage}."
            echo "[logs] ${LOG_DIR}"
            pause_if_needed
            return "${status}"
          fi
        fi
      done
    done
  done

  echo
  echo "[summary] Finished with ${failures} failed stage(s)."
  echo "[logs] ${LOG_DIR}"
  pause_if_needed

  if [[ "${failures}" -gt 0 ]]; then
    return 1
  fi
  return 0
}

main "$@"
