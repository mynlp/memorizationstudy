#!/bin/zsh
#PJM -L rscgrp=regular-o
#PJM -L node=25
#PJM -L elapse=48:00:00
#PJM -g gk77
#PJM -j
#PJM -N deduped_merge_log
#PJM -o deduped_merge_log
#PJM -e deduped_merge_log

source /work/gk77/k77025/.zshrc
cd pythia
python3 utils/unshard_memmap.py --input_file "/work/gk77/k77025/memorizationstudy/undeduped_data/datasets--EleutherAI--pile-deduped-pythia-preshuffled/snapshots/4647773ea142ab1ff5694602fa104bbf49088408/document-00000-of-00020.bin" --num_shards 21 --output_dir "/work/gk77/k77025/memorizationstudy/undeduped_merge"
