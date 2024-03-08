#!/bin/zsh
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:10:00
#PJM --mpi proc=2
#PJM -g gk77
#PJM -j
#PJM -N de1b3216
#PJM -o de1b3216
#PJM -e de1b3216
#if [ -z "$RUN_ON_REMOTE" ]; then
#    source /work/gk77/k77025/.zshrc
#fi
#source /work/gk77/k77025/.zshrc
MODULES_INIT_SCRIPT="/usr/share/Modules/init/bash"

source /usr/share/Modules/init/zsh
module load gcc/8.3.1
module load ompi/4.1.modu1



mpirun -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -map-by node ./paralle.sh