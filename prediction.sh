#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=48:00:00
#PJM -g gk77
#PJM -j
#PJM -N pred1b16
#PJM -o pred1b16
#PJM -e pred1b16

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python prediction.py --model_size 1b --context_size 32 --continuation_size 16
#python batch_generate.py