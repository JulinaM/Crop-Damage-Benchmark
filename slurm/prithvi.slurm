#!/bin/bash

#SBATCH --job-name=Prithvi
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
# Minimal (default): sbatch slurm/prithvi.slurm
# Debug infra logs: sbatch --export=DATA_SIZE=large,SLURM_LOG_LEVEL=debug slurm/prithvi.slurm

DATA_SIZE=${DATA_SIZE:-large} #large|small
SLURM_LOG_LEVEL="${SLURM_LOG_LEVEL:-minimal}"  # minimal|debug
PROJ_DIR="/users/PGS0218/julina/projects/geography/Crop-Damage-Benchmark/"
LOG_DIR="${PROJ_DIR}/data/slurm_logs"
LOG_STEM="prithvi-${DATA_SIZE//_/-}" ;;

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


log ">> Job start | id=${SLURM_JOB_ID} type=${DATA_SIZE} host=$(hostname)"

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
    CMD=(
      python3 -u -m crop_damage "--config-name=prithvi_large"
    )
    ;;
  small)
    CMD=(
      python3 -u -m crop_damage "--config-name=prithvi_small"
    )
    ;;
  *)
    >&2 echo "ERROR: Invalid job type '${DATA_SIZE}'. Valid options: {large|small}"
    exit 2
    ;;
esac

if [[ "${SLURM_LOG_LEVEL}" == "debug" ]]; then
  log "cmd=${CMD[*]}"
fi
# srun --job-name="TM_${DATA_SIZE}" \
"${CMD[@]}"

echo
echo "====== Job id=${SLURM_JOB_ID}  '$DATA_SIZE' finished successfully. ======"
