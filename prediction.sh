#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -g gk77
#PJM -j
#PJM -N pred41048
#PJM -o pred41048
#PJM -e pred41048

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python prediction.py --model_size 410m --context_size 32 --continuation_size 48
#python batch_generate.py