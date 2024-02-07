#PJM -L rscgrp=share-debug
#PJM -L gpu=2
#PJM -L elapse=00:01:00
#PJM -g gk77
#PJM -j
#PJM -N test
#PJM -o test
#PJM -e test

env
#export RANK=1
#export MODEL=70m-deduped-v0
#export CHECKPOINT=143000
#
#
#python3 generate.py