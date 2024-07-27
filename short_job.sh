#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -g gk77
#PJM -j
#PJM -N short_job
#PJM -o short_job
#PJM -e short_job

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
#python prediction_inference.py
#python distribution_analysis.py
#python batch_generate.py
#python embedding_drift.py
python corpus_sankey_flow.py