#!/bin/bash

#SBATCH --job-name=Terramind
#SBATCH --account=PGS0413
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -Eeuo pipefail

## USAGE::
# Minimal (default): sbatch slurm/terramind.slurm  # task=seg dataset=f -> segmentation/terramind_flood
# Choose task/dataset: sbatch --export=TASK=cd,DATASET=b slurm/terramind.slurm
# Quick smoke test: sbatch --export=DATA_SIZE=test slurm/terramind.slurm
# Debug infra logs: sbatch --export=DATA_SIZE=test,SLURM_LOG_LEVEL=debug slurm/terramind.slurm

DATA_SIZE=${DATA_SIZE:-large} #large|test
TASK="${TASK:-seg}"       # seg|cd
DATASET="${DATASET:-f}"   # f|b|p
SLURM_LOG_LEVEL="${SLURM_LOG_LEVEL:-minimal}"  # minimal|debug
PROJ_DIR="/users/PGS0218/julina/projects/geography/Crop-Damage-Benchmark/"
LOG_DIR="${PROJ_DIR}/data/slurm_logs"
if [[ "${DATA_SIZE}" == "test" ]]; then
  LOG_STEM="terramind-test"
else
  LOG_STEM="terramind-${TASK}-${DATASET}"
fi

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/${LOG_STEM}_${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/${LOG_STEM}_${SLURM_JOB_ID}.err"

timestamp() { date +"%Y-%m-%d %H:%M:%S %Z"; }
log() { echo "[SLURM $(timestamp)] $*"; }

on_error() {
  local exit_code=$?
  >&2 echo "[SLURM $(timestamp)] ERROR: command failed (exit=${exit_code}) at line ${BASH_LINENO[0]}"
  >&2 echo "[SLURM $(timestamp)] ERROR: ${BASH_COMMAND}"
  >&2 echo "[SLURM $(timestamp)] DATA_SIZE '${DATA_SIZE}' failed"
  exit "${exit_code}"
}
trap on_error ERR

# Reproducible environment setup.
cd "${PROJ_DIR}"
module --force purge
module load cuda/12.4.1 miniconda3/24.1.2-py310
source activate lora


log ">> Job start | id=${SLURM_JOB_ID} type=${DATA_SIZE} task=${TASK} dataset=${DATASET} host=$(hostname)"

if [[ "${SLURM_LOG_LEVEL}" == "debug" ]]; then
  echo
  log "pwd=$(pwd)"
  log "python=$(which python3)"
  nvidia-smi
fi

# Better traceback formatting for Hydra exceptions.
export HYDRA_FULL_ERROR=1

case $DATA_SIZE in
  large)
    case $TASK in
      seg) TASK_DIR="segmentation" ;;
      cd)  TASK_DIR="change_detection" ;;
      *)
        >&2 echo "ERROR: Invalid TASK '${TASK}'. Valid options: {seg|cd}"
        exit 2
        ;;
    esac

    case $DATASET in
      f) DATASET_NAME="flood" ;;
      b) DATASET_NAME="burnt" ;;
      p) DATASET_NAME="pooled" ;;
      *)
        >&2 echo "ERROR: Invalid DATASET '${DATASET}'. Valid options: {f|b|p}"
        exit 2
        ;;
    esac

    CONFIG_NAME="${TASK_DIR}/terramind_${DATASET_NAME}"
    if [[ ! -f "${PROJ_DIR}/configs/${CONFIG_NAME}.yaml" ]]; then
      >&2 echo "ERROR: Config 'configs/${CONFIG_NAME}.yaml' does not exist (TASK=${TASK} not yet implemented for change_detection? see configs/change_detection/README.md)."
      exit 2
    fi

    CMD=(
      python3 -u -m crop_damage "--config-name=${CONFIG_NAME}"
    )
    ;;
  test)
    CMD=(
      python3 -u -m crop_damage "--config-name=terramind_test"
    )
    ;;
  *)
    >&2 echo "ERROR: Invalid job type '${DATA_SIZE}'. Valid options: {large|test}"
    exit 2
    ;;
esac

if [[ "${SLURM_LOG_LEVEL}" == "debug" ]]; then
  log "cmd=${CMD[*]}"
fi
# srun --job-name="TM_${DATA_SIZE}" \
"${CMD[@]}"

echo
echo "====== Job id=${SLURM_JOB_ID}  '${DATA_SIZE}' (task=${TASK} dataset=${DATASET}) finished successfully. ======"
