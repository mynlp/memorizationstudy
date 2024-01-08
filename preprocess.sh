#!/bin/zsh
#PJM -L rscgrp=regular-o
#PJM -L node=40
#PJM -L elapse=48:00:00
#PJM -g gk77
#PJM -j
#PJM -N preprocess
#PJM -o preprocess
#PJM -e preprocess

if [ -z "$RUN_ON_REMOTE" ]; then
    source /work/gk77/k77025/.zshrc
fi
python3 utils/unshard_memmap.py --input_file "/work/gk77/k77025/memorizationstudy/deduped_data/datasets--EleutherAI--pile-standard-pythia-preshuffled/snapshots/bac79b6820adb34e451f9a02cc1dc7cd920febf0/document-00000-of-00020.bin" --num_shards 21 --output_dir "/work/gk77/k77025/memorizationstudy/deduped_merge"