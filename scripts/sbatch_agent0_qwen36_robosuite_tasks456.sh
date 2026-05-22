#!/bin/bash
# Sbatch wrapper: cap-x agent0 Robosuite eval on tasks 4/5/6
# (nut_assembly, spill_wipe, two_arm_lift) against a Qwen3.6-27B vLLM
# server that is already RUNNING on a separate node.
#
# Submit from login node:
#   sbatch scripts/sbatch_agent0_qwen36_robosuite_tasks456.sh
#
# Prereq: `sbatch qwen36_27b/serve_qwen36.sh` already RUNNING.
#
#SBATCH --job-name=capx-eval-tasks456
#SBATCH --partition=boost_usr_prod
#SBATCH --account=euhpc_d33_222
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd /leonardo/home/userexternal/zwang003/Projects/cap-x
mkdir -p logs

echo "=== Slurm ==="
echo "  job:  ${SLURM_JOB_ID:-?}"
echo "  node: $(hostname)"
echo "  gpus: ${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi -L || true

module purge
module load cuda/12.6
module load gcc/12.2.0

# Pin to the already-running Qwen server.
export QWEN_HOST="${QWEN_HOST:-lrdn3347}"
export QWEN_PORT="${QWEN_PORT:-8000}"
export QWEN_MODEL="${QWEN_MODEL:-Qwen3.6-27B}"

# Custom-mode inputs for run_agent0_qwen36_robosuite.sh.
export CAPX_CONFIGS="\
env_configs/nut_assembly/franka_robosuite_nut_assembly_multiturn_vdm_reduced_api_skill_lib.yaml \
env_configs/spill_wipe/franka_robosuite_spill_wipe_multiturn_vdm_reduced_api_skill_lib.yaml \
env_configs/two_arm_lift/franka_robosuite_two_arm_lift_multiturn_vdm_reduced_api_skill_lib.yaml"
export CAPX_OUTPUT_SUFFIX="tasks456_2026-05-17"
export CAPX_TOTAL_TRIALS=""   # empty → YAML default (100)
export NUM_WORKERS=8

exec bash scripts/run_agent0_qwen36_robosuite.sh custom
