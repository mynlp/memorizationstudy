#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -g gk77
#PJM -j
#PJM -N findacross12b
#PJM -o findacross12b
#PJM -e findacross12b

#env
#export RANK=1pjs
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
python find_cross.py --model_size 410m  --batch_size 50 --context_size 32 --continuation_size 16 --num_samples 2000
