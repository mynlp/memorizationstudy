#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -L elapse=48:00:00
#PJM -g gk77
#PJM -j
#PJM -N find70m3232
#PJM -o find70m3232
#PJM -e find70m3232

#env
#export RANK=1pjs
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
python find_cross.py --model_size 70m  --batch_size 5 --context_size 32 --continuation_size 32 --num_samples 2000
