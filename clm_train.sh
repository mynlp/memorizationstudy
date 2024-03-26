#!/bin/zsh
#PJM -L rscgrp=share-debug
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -g gk77
#PJM -j
#PJM -N clm
#PJM -o clm
#PJM -e clm

#env
#export RANK=1pjs
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
python clmtraining.py