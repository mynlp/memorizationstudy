#!/bin/zsh
#PJM -L rscgrp=regular-a
#PJM -L node=2
#PJM -L elapse=1:00:00
#PJM --mpi proc=2
#PJM -g gk77
#PJM -j
#PJM -N de1b3216
#PJM -o de1b3216
#PJM -e de1b3216

module load gcc/8.3.1
module load ompi/4.1.1

mpirun -machinefile $PJM_O_NODEINF -np $PJM_MPI_PROC -map-by node ./paralle.sh