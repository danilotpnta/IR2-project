#!/bin/bash

#SBATCH --job-name=trec-covid_cot_end_to_end
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/scur2880/IR2-project/logs/%x_%A.out


date

# Environment setup
module purge
module load 2024
module load Java/21.0.2

export HF_HOME="/scratch-shared/$USER/.cache/huggingface"

## Runs
# Estimated times for each step:
#   - Filter:      30 min
#   - Gentriples:  30 min
#   - Train:        3 min
#   - Rerank:       6 h
#   - Evaluate:    30 min

###
# DATASET="scifact" 
# STRATEGY="Zero-shot"

# DATASET="scifact" 
# STRATEGY="CoT"

###
# DATASET="trec-covid" 
# STRATEGY="Zero-shot"

DATASET="trec-covid" 
STRATEGY="CoT"

###
# DATASET="arguana" 
# STRATEGY="Zero-shot"

# DATASET="arguana" 
# STRATEGY="CoT"

###
# DATASET="nfcorpus" 
# STRATEGY="Zero-shot"

# DATASET="nfcorpus" 
# STRATEGY="CoT"



## Variables
PROJECT_ROOT="/home/$USER/IR2-project"
RESULTS_DIR="$PROJECT_ROOT/results/$DATASET"
SCRATCH_RERANKERS="/scratch-shared/$USER/rerankers/$DATASET"
MODEL="Meta-Llama-3.1-Instruct-8B_merged-16bit_CPO_MSMARCO" # "Llama-3.1-8B"
SUBSET="" # "_10k"


cd "$PROJECT_ROOT"
source IR2-env/bin/activate


srun python -m inpars.filter \
        --input="results/$DATASET/queries_${MODEL}_${STRATEGY}${SUBSET}.jsonl" \
        --dataset="$DATASET" \
        --filter_strategy="reranker" \
        --keep_top_k="10_000" \
        --output="$RESULTS_DIR/queries_${MODEL}_${STRATEGY}_filtered${SUBSET}.jsonl" \
        --use_scratch_shared_cache \
        --keep_only_question \
        --batch_size 32 \
        --fp16
echo -e "** Finished filtering! **\n"

sleep 10
srun python -m inpars.generate_triples \
        --input="$RESULTS_DIR/queries_${MODEL}_${STRATEGY}_filtered${SUBSET}.jsonl" \
        --dataset="$DATASET" \
        --output="$RESULTS_DIR/queries_${MODEL}_${STRATEGY}_triplets${SUBSET}.tsv"
echo -e "** Finished generating triples! **\n"

sleep 10
srun python -m inpars.train \
        --triples="$RESULTS_DIR/queries_${MODEL}_${STRATEGY}_triplets${SUBSET}.tsv" \
        --base_model="castorini/monot5-3b-msmarco-10k" \
        --output_dir="$SCRATCH_RERANKERS/${STRATEGY}/${SUBSET}" \
        --max_steps="156"
echo -e "** Finished training! **\n"

sleep 10
srun python -m inpars.rerank \
        --model="$SCRATCH_RERANKERS/${STRATEGY}/${SUBSET}" \
        --dataset="$DATASET" \
        --output_run="$RESULTS_DIR/queries_${MODEL}_${STRATEGY}${SUBSET}.txt" \
        --batch_size 64 \
        --fp16
echo -e "** Finished reranking! **\n"

sleep 10
srun python -m inpars.evaluate \
        --dataset="$DATASET" \
        --run="$RESULTS_DIR/queries_${MODEL}_${STRATEGY}${SUBSET}.txt" \
        --json \
        --output_path="$RESULTS_DIR/queries_${MODEL}_${STRATEGY}_results${SUBSET}.json"
echo -e "** Finished evaluation! **\n"
