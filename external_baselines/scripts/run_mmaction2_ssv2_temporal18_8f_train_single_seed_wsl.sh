#!/usr/bin/env bash
# Run MMAction2 8-frame baselines on SSv2-Temporal18 for one seed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTERNAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${EXTERNAL_ROOT}/.." && pwd)"
BASELINE_FILE="${EXTERNAL_ROOT}/experiments/baselines_ssv2_temporal18.yaml"

CONDA_ENV="${CONDA_ENV:-retail_mmaction2}"
CONFIG_PYTHON="${CONFIG_PYTHON:-python}"
SEED="${SEED:-1024}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${EXTERNAL_ROOT}/run_logs/batch/mmaction2_ssv2_temporal18_8f_train_seed${SEED}_${RUN_ID}"

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

  if [[ "${#MODELS[@]}" -eq 0 ]]; then
    echo "[error] No MMAction2 8-frame models found in ${BASELINE_FILE}."
    return 2
  fi

  echo "[project] ${PROJECT_ROOT}"
  echo "[baseline] ${BASELINE_FILE}"
  echo "[logs] ${LOG_DIR}"
  echo "[conda env] ${CONDA_ENV}"
  echo "[models] ${MODELS[*]}"
  echo "[seed] ${SEED}"

  local failures=0
  local model=""
  local status=0
  local cmd=()

  for model in "${MODELS[@]}"; do
    cmd=(
      conda run --no-capture-output -n "${CONDA_ENV}" python
      external_baselines/scripts/run_experiment.py
      --experiment-file "${BASELINE_FILE}"
      --model "${model}"
      --seed "${SEED}"
      --stage train
    )

    run_and_log "${model}_seed${SEED}_train" "${cmd[@]}"
    status=$?
    if [[ "${status}" -ne 0 ]]; then
      failures=$((failures + 1))
    fi
  done

  echo
  echo "[summary] Finished with ${failures} failed train run(s)."
  echo "[logs] ${LOG_DIR}"
  return 0
}

main "$@"
