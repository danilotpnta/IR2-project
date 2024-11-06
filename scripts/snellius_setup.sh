#!/bin/bash

setup() {
	module purge
	module load 2024
	module load Python/3.12.3-GCCcore-13.3.0

	source "$1/.venv/bin/activate"
}

