#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=8
#PJM -L elapse=24:00:00
#PJM --mpi proc=8
#PJM -g gk77
#PJM -j
#PJM -N de12bm3232
#PJM -o de12bm3232
#PJM -e de12bm3232
#if [ -z "$RUN_ON_REMOTE" ]; then
#    source /work/gk77/k77025/.zshrc
#fi
#source /work/gk77/k77025/.zshrc
#MODULES_INIT_SCRIPT="/usr/share/Modules/init/bash"

source /usr/share/Modules/init/zsh
module load gcc/8.3.1
module load ompi/4.1.1
## available models: 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b deduped
## available models: 14m, 31m, 70m, 160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b undeduped
echo $PJM_O_NODEINF >> /work/gk77/share/log
cat $PJM_O_NODEINF >> /work/gk77/share/log
#N_NODE=$1
#RANK=$2
#CONTEXT_SIZE=$3
#CONTINUATION_SIZE=$4
#BATCH_SIZE=$5
#MODEL=$6
mpirun -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -map-by node ./paralle.sh 8 8 32 32 1024 6.9b-deduped-v0