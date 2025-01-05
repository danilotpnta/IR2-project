#!/bin/bash

#SBATCH --partition=gpu_h100
#SBATCH --gpus=1

#SBATCH --job-name=DSPy_end_to_end
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:59:00
#SBATCH --output=/scratch-shared/scur2880/logs/scifact_Agent_DSPy_end_to_end_%A.out

#SBATCH --ear=on
#SBATCH --ear-policy=monitoring
#SBATCH --ear-verbose=1

date

# Environment setup
module purge
module load 2024
module load Java/21.0.2

export HF_HOME="/scratch-shared/$USER/.cache/huggingface"

# Variables
DATASET="scifact"
# STRATEGY="Zero-shot"
# STRATEGY="CoT"
STRATEGY="Agent"
PROJECT_ROOT="/home/$USER/IR2-project"
RESULTS_DIR="$PROJECT_ROOT/results/$DATASET"
SCRATCH_RERANKERS="/scratch-shared/$USER/rerankers/$DATASET"

cd "$PROJECT_ROOT"
source IR2-env/bin/activate

# Estimated times for each step:
#   - Filter:      30 min
#   - Gentriples:  30 min
#   - Train:        3 min
#   - Rerank:       6 h
#   - Evaluate:    30 min

srun python -m inpars.filter \
        --input="data/$DATASET/queries_Llama-3.1-8B_${STRATEGY}.jsonl" \
        --dataset="$DATASET" \
        --filter_strategy="reranker" \
        --keep_top_k="10_000" \
        --output="$RESULTS_DIR/queries_Llama-3.1-8B_${STRATEGY}_filtered.jsonl" \
        --use_scratch_shared_cache \
        --keep_only_question \
        --batch_size 32 \
        --fp16

sleep 120
srun python -m inpars.generate_triples \
        --input="$RESULTS_DIR/queries_Llama-3.1-8B_${STRATEGY}_filtered.jsonl" \
        --dataset="$DATASET" \
        --output="$RESULTS_DIR/queries_Llama-3.1-8B_${STRATEGY}_triplets.tsv"

sleep 120
srun python -m inpars.train \
        --triples="$RESULTS_DIR/queries_Llama-3.1-8B_${STRATEGY}_triplets.tsv" \
        --base_model="castorini/monot5-3b-msmarco-10k" \
        --output_dir="$SCRATCH_RERANKERS/${STRATEGY}/" \
        --max_steps="156"

sleep 120
srun python -m inpars.rerank \
        --model="$SCRATCH_RERANKERS/${STRATEGY}/" \
        --dataset="$DATASET" \
        --output_run="$RESULTS_DIR/queries_Llama-3.1-8B_${STRATEGY}.txt" \
        --batch_size 64 \
        --fp16

sleep 120
srun python -m inpars.evaluate \
        --dataset="$DATASET" \
        --run="$RESULTS_DIR/queries_Llama-3.1-8B_${STRATEGY}.txt" \
        --json \
        --output_path="$RESULTS_DIR/queries_Llama-3.1-8B_${STRATEGY}_results.json"