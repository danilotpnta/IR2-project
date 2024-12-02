#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

#SBATCH --job-name=CPO_TRAIN
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:59:00
#SBATCH --output=/scratch-shared/scur2850/logs/cpo_train_%A.out

#SBATCH --ear=on
#SBATCH --ear-policy=monitoring
#SBATCH --ear-verbose=1

date
export HF_HOME="/scratch-shared/$USER"
export GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)

# make sure the correct modules are used and that the virtual environment is active
PROJECT_ROOT=/home/$USER/IR2-project
source $PROJECT_ROOT/scripts/snellius_setup.sh
setup $PROJECT_ROOT
cd $PROJECT_ROOT

ARGS_PATH="$PROJECT_ROOT/config/cpo_args.json"

# rest of the script
srun python -m inpars.cpo_finetune $ARGS_PATH
