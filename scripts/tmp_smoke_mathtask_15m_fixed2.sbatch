#!/bin/bash
#SBATCH --job-name=smk_mathtask
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH -o /scratch/u6g/zhouyihe.u6g/verl_v0.4.x/slurm_logs/%x_%j.out
set -euo pipefail
cd /scratch/u6g/zhouyihe.u6g/verl_v0.4.x
apptainer exec --nv -B /scratch/u6g/zhouyihe.u6g:/mnt/home -B /dev/shm:/dev/shm /scratch/u6g/zhouyihe.u6g/sglang.sif bash -lc '
  set -euo pipefail
  cd /mnt/home/verl_v0.4.x
  export PYTHON_BIN=/mnt/home/verl_v0.4.x/.verl/bin/python
  export CACHE_BASE=/mnt/home/verl_cache/${SLURM_JOB_ID}
  export RAY_TMPDIR=/mnt/home/raytmp/${SLURM_JOB_ID}
  export HOME=/mnt/home/verl_cache/${SLURM_JOB_ID}/home
  export XDG_CACHE_HOME=/mnt/home/verl_cache/${SLURM_JOB_ID}/xdg_cache
  mkdir -p ${HOME} ${XDG_CACHE_HOME} ${RAY_TMPDIR}
  export NGPUS_PER_NODE=2
  export NNODES=1
  export TEST_FREQ=1
  export TOTAL_EPOCHS=1
  export TRAIN_PROMPT_BSZ=32
  export TRAIN_PROMPT_MINI_BSZ=8
  export MICRO_BATCH_SIZE_PER_GPU=4
  export VAL_N=2
  bash examples/RL_math/run_qwen2.5-3b_math_grpo.sh Qwen/Qwen2.5-3B-Instruct trainer.logger=["console"]
'
