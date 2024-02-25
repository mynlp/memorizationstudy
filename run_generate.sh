#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -L elapse=48:00:00
#PJM -g gk77
#PJM -j
#PJM -N de1b3296
#PJM -o de1b3296
#PJM -e de1b3296

#env
#export RANK=1
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

## available models: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b deduped
## available models: 14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b undeduped
python3 batch_generate.py --model 1b-deduped-v0 --checkpoint 143000 --batch_size 1024 --context_size 32 --continuation_size 96


#python3 batch_generate.py --model 1b-deduped-v0 --checkpoint 143000 --batch_size 1024 --context_size 32 --continuation_size 16