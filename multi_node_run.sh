#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=8
#PJM -L elapse=24:00:00
#PJM --mpi proc=8
#PJM -g gk77
#PJM -j
#PJM -N de6.9b3216
#PJM -o de6.9b3216
#PJM -e de6.9b3216
#if [ -z "$RUN_ON_REMOTE" ]; then
#    source /work/gk77/k77025/.zshrc
#fi
#source /work/gk77/k77025/.zshrc
#MODULES_INIT_SCRIPT="/usr/share/Modules/init/bash"

source /usr/share/Modules/init/zsh
module load gcc/8.3.1
module load ompi/4.1.1

echo $PJM_O_NODEINF >> /work/gk77/share/log
cat $PJM_O_NODEINF >> /work/gk77/share/log
#N_NODE=$1
#RANK=$2
#CONTEXT_SIZE=$3
#CONTINUATION_SIZE=$4
#BATCH_SIZE=$5
#MODEL=$6
mpirun -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -map-by node ./paralle.sh 8 8 32 16 512 6.9b-deduped-v0