#!/bin/bash

setup() {
	module purge
	module load 2024
	module load Java/21.0.2

	source "$1/.venv/bin/activate"

	echo "Environment setup complete."
	echo "Python version: $(python --version)"
	echo "Java version: $(java --version)"
	echo "Java compiler version: $(javac --version)"
}