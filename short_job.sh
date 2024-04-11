#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=00:30:00
#PJM -g gk77
#PJM -j
#PJM -N shortjob
#PJM -o shortjob
#PJM -e shortjob

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi

python distribution_analysis.py