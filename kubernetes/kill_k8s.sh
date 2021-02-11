#!/usr/bin/env bash

#  --- USAGE ---
# $ kill_k8s.sh [name_suffix=$USER]

SUFFIX=${1:-$(whoami)}
DEPLOYMENT_NM='megatron-'"$SUFFIX"

kubectl delete deploy/$DEPLOYMENT_NM
