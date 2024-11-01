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
conda create -n ir2-project python=3.10
conda activate ir2-project
```

### Step 3: Install the Required Packages
```bash
conda upgrade pip
conda install -c conda-forge nmslib
pip install -r requirements.txt
```
### Generating the Synthetic Data
By default when you generate the synthetic data, the InPars Toolkit will download (if not already done) one of the BEIR datasets at `~/.ir_datasets/beir/` directory. 


### Troubleshooting

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.