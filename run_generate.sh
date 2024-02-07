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

python3 batch_generate.py