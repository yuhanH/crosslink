#!/bin/bash

# Define the number of nodes and GPUs per node
NUM_NODES=1
GPUS_PER_NODE=2

# Define the command to run on each node
CMD = "python main_dist.py " \
      "--split tf_in_domain " \
      "--run_dir /home/ubuntu/codebase/tf_binding/runs/ " \
      "--epoches 200 " \
      "--batch_size 128"
# Define the master node
MASTER_NODE=0

# Define the port number
PORT=1234

# Launch the distributed training
torchrun --nnodes=$NUM_NODES --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$MASTER_NODE --master_addr="node-${MASTER_NODE}" \
    --master_port=$PORT $CMD