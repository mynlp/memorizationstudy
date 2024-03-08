#!/bin/zsh

python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=`hostname -i` --master_port=29501 distributed_generate.py

#torchrun --nproc_per_node=$OMPI_COMM_WORLD_RANK distributed_generate.py --batch_size=1024 --context_size=$CONTEXT_SIZE --continuation_size=$CONTINUATION_SIZE --model=$MODEL --checkpoint=143000
