#!/bin/bash

setup() {
	module purge
	module load 2022
	#module load Python/3.12.3-GCCcore-13.3.0

	# 3.10 necessary for pyserini
	module load Python/3.10.4-GCCcore-11.3.0

	# Java>.16 necessary dependency, so need to load 2024 as well
	module load 2024
	module load Java/21.0.2

	source "$1/.venv/bin/activate"

	export CUDA_VISIBLE_DEVICES=0
	export TF_ENABLE_ONEDNN_OPTS=0
	export WANDB_API_KEY=

	echo "Environment setup complete."
	echo "Python version: $(python --version)"
	echo "Java version: $(java --version)"
	echo "Java compiler version: $(javac --version)"
}
