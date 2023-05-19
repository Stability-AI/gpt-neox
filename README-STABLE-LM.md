# stableLLM

## Description

This repository contains the code for training and evaluating the stableLLM model. It utilizes the neox library for the GPT-Neo model.

## Installation

To install this repository, first clone the repository and its submodules:

```bash
git clone --recursive https://github.com/Stability-AI/gpt-neox.git
```

Setup the environment:

For SLURM (optional):
```bash
srun --exclusive --account=stablegpt --nodes=1 --partition=g40 --pty bash -i
source /etc/profile.d/modules.sh
module load cuda/11.7
```

Virtual Environment:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py38_23.3.1-0-Linux-x86_64.sh
```
    
```bash
source ~/.bashrc
conda env create -f env.yaml
conda activate stable-lm
```

Next install the third party libraries:

```bash
bash setup.sh
```

## Test Installation

To test the installation, run the following command:

```bash
bash test.sh
```

This test runs the model on a small dataset and checks that the output is correct.

## Run Training

To run a complete training run, use the provided `deploy.sh` script that uses slurm to configure everything:

```bash
bash deploy.sh -n <num_nodes> -c <config_path> -j <job_name> 

# Or using long form arg names:

bash deploy.sh --nodes <num_nodes> --config <config_path> --jobname <job_name>
```

If you want to test a single node training with sbatch, you can use the following command:

```bash
bash deploy.sh -c ./configs/test_config -j single_node_test -n 1
```

To perform a multi-node training run, you can use the following command:

```bash
bash deploy.sh --config ./configs/test_multi_node_config --jobname multi_node_test --nodes 2
```

For multinode, ensure that your config file has the following settings:

```bash
# gpt-neox/configs/config.yml
"launcher": "openmpi",
"deepspeed_mpi": true,
```

or

```bash
# gpt-neox/configs/config.yml
"launcher": "slurm",
"deepspeed_slurm": true,
```

Depending on the launcher you want to use.
