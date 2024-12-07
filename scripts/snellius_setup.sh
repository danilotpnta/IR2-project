#!/bin/bash

setup() {
	module purge
	
	module load 2024
	module load Java/21.0.2
    module load CUDA/12.6.0

	source "$1/.venv/bin/activate"

	# export CUDA_VISIBLE_DEVICES=0
	export TF_ENABLE_ONEDNN_OPTS=0
	# export WANDB_API_KEY=
	# export HF_TOKEN=
	
	# The pretrained MonoT5 models are >22GB, so cache in scratch-shared
	export HF_HOME="/scratch-shared/$USER"

	echo "Environment setup complete."
	echo "Python version: $(python --version)"
	echo "Java version: $(java --version)"
	echo "Java compiler version: $(javac --version)"
	huggingface-cli login --token $HF_ACCESS_TOKEN
}
