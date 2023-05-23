#!/bin/bash

usage()
{
  echo "Usage: bash deploy.sh [ -c | --config ] [ -j | --jobname ] [ -n | --nodes ]"
  exit 2
}

PARSED_ARGUMENTS=$(getopt -o c:j:n: --long config:,jobname:,nodes: -- "$@")
echo "PARSED_ARGUMENTS is $PARSED_ARGUMENTS"
eval set -- "$PARSED_ARGUMENTS"

while true ; do
    case "$1" in
        -c|--config) config=$2 ; shift 2 ;;
        -j|--jobname) jobname=$2 ; shift 2 ;;
        -n|--nodes) nodes=$2 ; shift 2 ;;
        --) shift; break ;;
        *) echo "Unexpected option: $1 - this should not happen."
        usage ;;
    esac
done

mkdir -p sbatches
cat << EOF > sbatches/sbatch_runner_$jobname.sh
#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name=${jobname}
#SBATCH --partition=g40
#SBATCH --nodes=$nodes # Set > 1 for multi-node
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=11G
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

KILLED=137
TERMINATED=143
ABORTED=134

REPEAT_COUNTER=\${1:-0}
MAX_RUNS=2

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
export NCCL_DEBUG=warn
# Uncomment the following lines to enable detailed NCCL debug logging
# CWD=\$(pwd)
# mkdir -p \$CWD/nccl-logs/\$SLURM_JOB_ID  
# export NCCL_DEBUG_FILE=\$CWD/nccl-logs/\$SLURM_JOB_ID/debug.%h.%p
# export NCCL_DEBUG_SUBSYS=all  # Allows filtering the NCCL_DEBUG=INFO output based on subsystems
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

echo "$0 = \$0"
bash -c 'python ./deepy.py train.py ${config}; exit \$?'
RETVAL=\$?
echo "RETVAL = \${RETVAL}"
# choose your action, we retry when process aborted,killed or signalled but not when it exited with 0 or non-zero code
# but only up to MAX_RUNS attempts
if [ \${RETVAL} -eq  \${ABORTED} -o \${RETVAL} -eq \${TERMINATED} -o \${RETVAL} -eq \${KILLED} ]
then
  let run=\${REPEAT_COUNTER}+1
  if [ \${run} -lt \${MAX_RUNS} ]
  then
    echo "Resubmitting job. Retry number = \${run}"
    sbatch \$0 \${run}
  fi
fi
EOF

sbatch sbatches/sbatch_runner_${jobname}.sh
