#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=32:00:00
#PJM -g gk77
#PJM -j
#PJM -N findacross
#PJM -o findacross
#PJM -e findacross

#env
#export RANK=1pjs
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
python find_cross.py
#python3 batch_generate.py --model 1b-deduped-v0 --checkpoint 143000 --batch_size 1024 --context_size 32 --continuation_size 16