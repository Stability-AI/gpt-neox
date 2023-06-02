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

mkdir -p sbatches logs
cat << EOF > sbatches/sbatch_runner_$jobname.sh
#!/bin/bash
#SBATCH --account="stablegpt"
#SBATCH --job-name=${jobname}
#SBATCH --partition=a100-cu117
#SBATCH --nodes=$nodes # Set > 1 for multi-node
#SBATCH --ntasks-per-node=8
#SBATCH --hint=nomultithread         # We get physical cores not logical
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

KILLED=137
TERMINATED=143
ABORTED=134

REPEAT_COUNTER=\${1:-0}
MAX_RUNS=2

###########################################################
# Pre-load
###########################################################
source /opt/hpcx/hpcx-init.sh
hpcx_load
###########################################################


###########################################################
# CUDA/Torch Setup
###########################################################
export NCCL_DEBUG=info
CWD=\$(pwd)
# Uncomment the following lines to enable detailed NCCL debug logging
# mkdir -p \$CWD/nccl-logs/\$SLURM_JOB_ID  
# export NCCL_DEBUG_FILE=\$CWD/nccl-logs/\$SLURM_JOB_ID/debug.%h.%p
# export NCCL_DEBUG_SUBSYS=all  # Allows filtering the NCCL_DEBUG=INFO output based on subsystems
export NCCL_TREE_THRESHOLD=0
export NCCL_PROTO=simple

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_EXTENSIONS_DIR=\$HOME/.cache/torch_extensions
# export TORCH_CPP_LOG_LEVEL=INFO
# export TORCH_DISTRIBUTED_DEBUG=INFO
###########################################################


###########################################################
# MPI Setup (if required)
###########################################################
# export OMPI_MCA_mtl_base_verbose=1
# export OMPI_MCA_btl="^openib"
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
bash \$CWD/scripts/write_hostfile.sh
hostfile=\$CWD/hostfiles/hosts_\$SLURM_JOBID
export DLTS_HOSTFILE=\$hostfile
echo "DLTS_HOSTFILE: \$DLTS_HOSTFILE"
###########################################################


###########################################################
# Environment Setup
# TODO: Replace with your own environment setup
###########################################################
source /mnt/nvme/home/$(whoami)/miniconda3/bin/activate stable-neox-env
###########################################################

ds_report

###########################################################
# NeoX Setup
###########################################################
export WANDB_MODE="online"
export WANDB_API_KEY=""
export PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256'

###########################################################
# Run launch vars Setup
###########################################################
export NEOX_WORKING_DIR=$(pwd)
export NEOX_LAUNCH_CMD="bash deploy_coreweave.sh -c ${config} -j ${jobname} -n ${nodes} "
export NEOX_LAUNCH_CONFIG_PATH=${config}
export NEOX_LAUNCH_JOB_NAME=${jobname}
export NEOX_LAUNCH_NODES=${nodes}

git config --global --add safe.directory \$NEOX_WORKING_DIR

echo "$0 = \$0"
echo "config = \$NEOX_LAUNCH_CONFIG_PATH"
wandb login --host https://stability.wandb.io --relogin \$WANDB_API_KEY
bash -c 'python deepy.py train.py ${config}; exit \$?'
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
