#!/bin/bash
mkdir -p $(pwd)/hostfiles
hostfile=$(pwd)/hostfiles/hosts_$SLURM_JOBID
rm $hostfile &> /dev/null  # for consecutive calls to this script in interactive mode
for i in $(scontrol show hostnames "$SLURM_JOB_NODELIST")
do
    export ip=`bash $(pwd)/scripts/extract_ip.sh $i`
    echo $i slots=8 >>$hostfile
done