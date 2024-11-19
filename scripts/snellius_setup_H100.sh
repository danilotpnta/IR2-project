#!/bin/bash

setup() {
	module purge
	
	# Setup when using H100 (necessary for finetuning MonoT5-3B , requires > 40GB)
	module load 2024
	module load Python/3.12.3-GCCcore-13.3.0
	module load Java/21.0.2
	source "$1/.venv312/bin/activate"

	export CUDA_VISIBLE_DEVICES=0
	export TF_ENABLE_ONEDNN_OPTS=0
	export WANDB_API_KEY=
	export HF_TOKEN=

	echo "Environment setup complete."
	echo "Python version: $(python --version)"
	echo "Java version: $(java --version)"
	echo "Java compiler version: $(javac --version)"
}
