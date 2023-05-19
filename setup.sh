#! /bin/bash

# Install dependencies
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
pip install -r requirements/requirements.txt
pip install megatron/fused_kernels