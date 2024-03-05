#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gk77
#PJM -j
#PJM -N de4103232
#PJM -o de4103232
#PJM -e de4103232

#env
#export RANK=1
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
RANK=8
CONTEXT_SIZE=32
CONTINUATION_SIZE=32
MODEL=1b-deduped-v0

## available models: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b deduped
## available models: 14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b undeduped
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node=$RANK distributed_generate.py --batch_size=1024 --context_size=$CONTEXT_SIZE --continuation_size=$CONTINUATION_SIZE --model=$MODEL --checkpoint=143000
python csv_merge.py --rank_size $RANK --checkpoint 143000 --context_size $CONTEXT_SIZE --continuation_size $CONTINUATION_SIZE --model $MODEL
#python3 batch_generate.py --model 1b-deduped-v0 --checkpoint 143000 --batch_size 1024 --context_size 32 --continuation_size 16