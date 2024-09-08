#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=2
#PJM -L elapse=24:00:00
#PJM -g gk77
#PJM -j
#PJM -N instrution
#PJM -o instrution
#PJM -e instrution

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python instrution_tuning_pythia.py