#!/bin/zsh

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
first_node=$(cat "$PJM_O_NODEINF" | awk 'NR==1{print $1}')
export MASTER_ADDR=$first_node
echo $MASTER_ADDR
#python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=`hostname -i` --master_port=29501 distributed_generate.py
cat $OMPI_COMM_WORLD_RANK >> /work/gk77/share/log
cat $MASTER_ADDR >> /work/gk77/share/log
#torchrun --nproc_per_node=$OMPI_COMM_WORLD_RANK distributed_generate.py --batch_size=1024 --context_size=$CONTEXT_SIZE --continuation_size=$CONTINUATION_SIZE --model=$MODEL --checkpoint=143000
torchrun --nproc_per_node=8 --nnodes=4 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR --master_port=29504 distributed_generate.py #--batch_size=1024 --context_size=$CONTEXT_SIZE --continuation_size=$CONTINUATION_SIZE --model=$MODEL --checkpoint=143000
