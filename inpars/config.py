from pathlib import Path
import os

# Root directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define directories based on ROOT_DIR
DATA_DIR = ROOT_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
