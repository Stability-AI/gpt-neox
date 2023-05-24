#!/bin/bash

mkdir sbatches
cat << EOF > sbatches/sbatch_runner_$1.sh
#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name="$1"
#SBATCH --partition=g40
#SBATCH --nodes=$2  # Set > 1 for multi-node
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/neox_%j.out
#SBATCH --error=logs/neox_%j.err
#SBATCH --exclude=ip-26-0-140-150,ip-26-0-134-43,ip-26-0-143-225,ip-26-0-135-173,ip-26-0-129-240,ip-26-0-140-63,ip-26-0-133-115,ip-26-0-131-[143,183,188,201],ip-26-0-132-214,ip-26-0-133-[81,126],ip-26-0-136-[27,42],ip-26-0-138-51,ip-26-0-140-[123-124],ip-26-0-143-[111,121,235,250],ip-26-0-129-197,ip-26-0-130-[37,127,132,150,164,193],ip-26-0-134-201,ip-26-0-137-[115,168,184,196],ip-26-0-138-208,ip-26-0-141-70,ip-26-0-142-[3,13],ip-26-0-130-[41,44],ip-26-0-131-[77,85,239-240],ip-26-0-133-[18,39,67,76],ip-26-0-134-[1,26-27],ip-26-0-136-[99,103,118]


KILLED=137
TERMINATED=143
ABORTED=134

REPEAT_COUNTER=\${1:-0}
MAX_RUNS=1

source /etc/profile.d/modules.sh
module load openmpi cuda/11.7

###########################################################
# Pre-load
###########################################################
HOME=/fsx/home-\$(whoami)
CUDNN_HOME=/fsx/quentin/cudnn-linux-x86_64-8.6.0.163_cuda11-archive
export LD_LIBRARY_PATH=\$CUDNN_HOME/lib:\$LD_LIBRARY_PATH
export CPATH=\$CUDNN_HOME/include:\$CPATH

CONDA_HOME=/fsx/home-\$(whoami)/miniconda3/envs/stable-neox-env
export PATH=\$CONDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CONDA_HOME/lib:\$LD_LIBRARY_PATH
export CPATH=\$CONDA_HOME/include:\$CPATH
###########################################################

###########################################################
# CUDA/Torch Setup
###########################################################
CWD=\$(pwd)
mkdir -p \$CWD/nccl-logs/\$SLURM_JOB_ID  
export NCCL_DEBUG_FILE=\$CWD/nccl-logs/\$SLURM_JOB_ID/debug.%h.%p
export NCCL_DEBUG=info
export NCCL_DEBUG_SUBSYS=all
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple
# Network issues without the following two NCCL vars set; See https://github.com/NVIDIA/nccl/issues/676
export NCCL_IBEXT_DISABLE=1
export NCCL_SOCKET_IFNAME=^docker0,lo

export CUDA_LAUNCH_BLOCKING=0
export TORCH_EXTENSIONS_DIR=\$HOME/.cache/torch_extensions
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
export PYTHONFAULTHANDLER=1
###########################################################


###########################################################
# FI Setup (if required)
###########################################################
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=warn
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export FI_EFA_USE_DEVICE_RDMA=-1 # use for p4dn
###########################################################


###########################################################
# MPI Setup (if required)
###########################################################
export OMPI_MCA_mtl_base_verbose=1
export OMPI_MCA_btl="^openib"
###########################################################


###########################################################
# Network Setup
###########################################################
export HOSTNAMES=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST")
export MASTER_ADDR=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=\$(scontrol show hostnames "\$SLURM_JOB_NODELIST" | wc -l)

echo "Master Addr: \$MASTER_ADDR"
echo "Node Count: \$COUNT_NODE"
echo "Host Names: \$HOSTNAMES"
lsof -i
cat /etc/hosts

# Write the hostfile for this job
export MASTER_ADDR=\$(echo \$MASTER_ADDR | cut -d '-' -f 2- | tr '-' '.')
bash \$CWD/scripts/write_hostfile.sh
export DLTS_HOSTFILE=\$CWD/hostfiles/hosts_\$SLURM_JOBID
###########################################################


sig_handler()
{
    echo "BATCH interrupted"
    wait # wait for all children, this is important!
}
trap 'sig_handler' SIGINT SIGTERM SIGCONT


###########################################################
# Environment Setup
# TODO: Replace with your own environment setup
###########################################################
source \$HOME/.bashrc
eval "\$(conda shell.bash hook)"
conda activate stable-neox-env
###########################################################

ds_report

###########################################################
# NeoX Setup
###########################################################
export WANDB_MODE="online"
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256'

TRAIN_PATH=\$CWD
cd \$TRAIN_PATH
wandb login --relogin --host https://stability.wandb.io local-edea3863613ef71e1ef7532673fdf4b46bc5ffd7
git config --global --add safe.directory \$TRAIN_PATH
python ./deepy.py train.py $3
RETVAL=\$?
echo "Test process returned \${RETVAL}"
# choose your action, we retry when process aborted,killed or signalled but not when it exited with 0 or non-zero code
# but only up to MAX_RUNS attempts
if [ \${RETVAL} -eq  \${ABORTED} -o \${RETVAL} -eq \${TERMINATED} -o \${RETVAL} -eq \${KILLED} ]
then
  let run=\${REPEAT_COUNTER}+1
  if [ \${run} -lt \${MAX_RUNS} ]
  then
    sbatch \$0 \${run}
  fi
fi
EOF

sbatch sbatches/sbatch_runner_$1.sh