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
conda env create -f environment.yml
conda activate IR2-env
```

## Running on Snellius

Follow Step 1 from the Installation section.

### Step 2: Create and Activate Python Environment

```bash
WORK_DIR=$HOME/IR2-project
cd $WORK_DIR

source scripts/snellius_setup.sh
setup $PWD
```

### Step 3: Install the Required Packages

```bash
# Install the required packages
pip install -e ".[all]"

# It might be faster to install using the requirements.txt file
pip install -r requirements.txt
```

### Troubleshooting

When installing in Snellius you may want to isntall the packages using the `--no-cache-dir` flag. This will prevent the installation from using the cache and may solve some issues.

```bash
pip install --no-cache-dir -r requirements.txt
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
