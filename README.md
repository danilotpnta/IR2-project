# IR2-project

Reproducibility Study of “InPars Toolkit: A Unified and Reproducible Synthetic Data Generation Pipeline for Neural Information Retrieval”

This project focuses on a reproducibility study of the InPars Toolkit, a tool designed for generating synthetic data to improve neural information retrieval (IR) systems. Our objective is to replicate and validate the methodology presented in the paper while improving on the future work proposed by the authors.

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/danilotpnta/IR2-project
cd IR2-project
```
### Step 2: Create and Activate the Conda Environment
```bash
conda create -n IR2-env python=3.10
conda activate IR2-env
```

### Step 3: Install the Required Packages
```bash
conda upgrade pip
conda install -c conda-forge nmslib
pip install -r requirements.txt
```

## Running on Snellius

Follow Step 1 from the Installation section.

### Step 2: Create and Activate Python Environment

```bash
WORK_DIR=$HOME/IR2-project
cd $WORK_DIR

# Loads Python 3.11 
module load 2023
module load Python/3.11.3-GCCcore-12.3.0  

# Creates a virtual environment
python3 -m venv IR2-env
module purge 
source IR2-env/bin/activate
```

### Step 3: Install the Required Packages
```bash
pip install --upgrade pip
pip install inpars
```

### Troubleshooting

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.