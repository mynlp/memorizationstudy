#!/bin/zsh
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=12:00:00
#PJM -g gk77
#PJM -j
#PJM -N distribution
#PJM -o distribution
#PJM -e distribution

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
#python prediction_inference.py
python distribution_analysis.py
#python batch_generate.py