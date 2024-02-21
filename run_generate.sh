#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=24:00:00
#PJM -g gk77
#PJM -j
#PJM -N test
#PJM -o test
#PJM -e test

#env
#export RANK=1
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

## available models:
python3 batch_generate.py --model 13m-deduped-v0 --checkpoint 143000 --batch_size 1024 --context_size 32 --continuation_size 32