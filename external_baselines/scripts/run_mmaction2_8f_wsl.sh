#!/usr/bin/env bash
# Run the fair-budget 8-frame MMAction2 external baselines on WSL.
#
# Default behavior:
# - reads model ids and seeds from external_baselines/experiments/baselines.yaml
# - selects all MMAction2 models whose frames field is 8
# - runs train, then val, then test for each model/seed
# - writes batch-level logs under external_baselines/run_logs/batch/
# - run_experiment.py writes stable stage logs under external_baselines/run_logs/<model>/seed<seed>/
# - stops on the first failure, prints the failing command, and pauses if interactive

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXTERNAL_ROOT}/.." && pwd)"
BASELINE_FILE="${EXTERNAL_ROOT}/experiments/baselines.yaml"

CONDA_ENV="${CONDA_ENV:-retail_mmaction2}"
CONFIG_PYTHON="${CONFIG_PYTHON:-python}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
PAUSE_ON_EXIT="${PAUSE_ON_EXIT:-1}"
EVAL_NUM_CROP="${EVAL_NUM_CROP:-3}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${EXTERNAL_ROOT}/run_logs/batch/mmaction2_8f_${RUN_ID}"

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

main() {
  mkdir -p "${LOG_DIR}"
  cd "${PROJECT_ROOT}"

  mapfile -t MODELS < <("${CONFIG_PYTHON}" -c "import json, sys; cfg=json.load(open(sys.argv[1], encoding='utf-8-sig')); print('\n'.join(m['model_id'] for m in cfg['models'] if m.get('backend') == 'mmaction2' and int(m.get('frames', -1)) == 8))" "${BASELINE_FILE}")
  mapfile -t SEEDS < <("${CONFIG_PYTHON}" -c "import json, sys; cfg=json.load(open(sys.argv[1], encoding='utf-8-sig')); print('\n'.join(str(s) for s in cfg['defaults'].get('seeds', [])))" "${BASELINE_FILE}")

  if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "[error] No MMAction2 8-frame models found in ${BASELINE_FILE}."
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
  echo "[eval num crop] ${EVAL_NUM_CROP}"
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
