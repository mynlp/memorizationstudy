#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -g gk77
#PJM -j
#PJM -N prediction410m
#PJM -o prediction410m
#PJM -e prediction1410m

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python prediction.py --model_size 410m --context_size 32 --continuation_size 64
#python batch_generate.py