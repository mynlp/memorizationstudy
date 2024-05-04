#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=24:00:00
#PJM -g gk77
#PJM -j
#PJM -N prediction6.9b
#PJM -o prediction6.9b
#PJM -e prediction6.9b

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python prediction.py --model_size 6.9b
#python batch_generate.py