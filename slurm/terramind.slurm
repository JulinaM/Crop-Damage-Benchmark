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
# Minimal (default): sbatch slurm/terramind.slurm
# Debug infra logs: sbatch --export=DATA_SIZE=curriculum_terramind,SLURM_LOG_LEVEL=debug slurm/terramind.slurm

DATA_SIZE=${DATA_SIZE:-large} #curriculum_terramind|large|small|large_search|small_search|old
SLURM_LOG_LEVEL="${SLURM_LOG_LEVEL:-minimal}"  # minimal|debug
PROJ_DIR="/users/PGS0218/julina/projects/geography/damage_mapping_terramind/V2"
LOG_DIR="${PROJ_DIR}/data/slurm_logs"

case "${DATA_SIZE}" in
  old) LOG_STEM="Terramind-train-old" ;;
  *) LOG_STEM="terramind-${DATA_SIZE//_/-}" ;;
esac

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
      python3 -u -m damage_mapping 
    )
    ;;
  curriculum_terramind)
    CMD=(
      python3 -u -m damage_mapping "--config-name=curriculum_terramind"
    )
    ;;
  small)
    CMD=(
      python3 -u -m damage_mapping "--config-name=terramind_small"
    )
    ;;
  
  large_search)
    CMD=(
      python3 -u -m damage_mapping "--config-name=terramind_large_search"
    )
    ;;
  small_search)
    CMD=(
      python3 -u -m damage_mapping "--config-name=terramind_small_search"
    )
    ;;

  old)
    CMD=(
      python3 -u -m damage_mapping 
      "model.modalities=[S2L2A]"
      "~train_loader.modalities.S1GRD"
      "~validation_loader.modalities.S1GRD"
      "model.num_classes=4"
    )
    ;;

  *)
    >&2 echo "ERROR: Invalid job type '${DATA_SIZE}'. Valid options: {large|small|large_search|small_search|old}"
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
