#!/bin/bash

setup() {
	module purge
	
	# Python 3.10 setup, for evaluation (pyserini needs 3.10)
	module load 2022
	module load Python/3.10.4-GCCcore-11.3.0
	module load 2024
	module load Java/21.0.2
	source "$1/.venv/bin/activate"

	export CUDA_VISIBLE_DEVICES=0
	export TF_ENABLE_ONEDNN_OPTS=0
	export WANDB_API_KEY=
	export HF_TOKEN=

	echo "Environment setup complete."
	echo "Python version: $(python --version)"
	echo "Java version: $(java --version)"
	echo "Java compiler version: $(javac --version)"
}
