from pathlib import Path
import os

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define directories based on ROOT_DIR
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"

MODEL_PORT_MAPPING = {
    "EleutherAI/gpt-j-6B": 8000,
    "meta-llama/Llama-3.1-8B": 8001,
    "neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic": 8002,
}

STOP_WORDS = ["\n", "\n\n", "Bad Question:", "Example", "Document:"]