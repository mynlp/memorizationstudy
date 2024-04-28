#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=12:00:00
#PJM -g gk77
#PJM -j
#PJM -N prediction
#PJM -o prediction
#PJM -e prediction

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python prediction.py
#python batch_generate.py