#!/bin/bash

# Define the number of nodes and GPUs per node
NUM_NODES=1
GPUS_PER_NODE=1

# Define the command to run on each node
CMD="main_dist.py --split chr --run_dir /home/ubuntu/codebase/tf_binding/runs/ --epoches 200 --batch_size 128"
# Define the master node

# Define the port number
PORT=1234

# Launch the distributed training
torchrun --nproc_per_node=$GPUS_PER_NODE $CMD
