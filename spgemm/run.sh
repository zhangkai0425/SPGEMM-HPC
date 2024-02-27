# !/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <executable>" >&2
  exit 1
fi

export DAPL_DBG_TYPE=0

DATAPATH=/home/2023-fall/course/hpc/assignments/2023-xue/mat_data
RESPATH=/home/2023-fall/course/hpc/assignments/2023-xue/gemm_res

EXECUTABLE=$1
REP=64

srun -n 1 ${EXECUTABLE} ${REP} ${DATAPATH}/utm5940.csr ${RESPATH}/utm5940.csr

# srun -n 1 ./benchmark-cusparse 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/ct2010.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/ct2010.csr
# srun -n 1 ./benchmark-optimized 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/ct2010.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/ct2010.csr

# srun -n 1 ./benchmark-optimized 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/utm5940.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/utm5940.csr

# srun -n 1 ./benchmark-optimized 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/fe_body.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/fe_body.csr
# srun -n 1 ./benchmark-cusparse 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/fe_body.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/fe_body.csr

# srun -n 1 ./benchmark-cusparse 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/nemeth07.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/nemeth07.csr

srun -n 1 ./benchmark-optimized 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/1138_bus.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/1138_bus.csr
srun -n 1 ./benchmark-optimized 64 /home/2023-fall/course/hpc/assignments/2023-xue/mat_data/lhr17.csr /home/2023-fall/course/hpc/assignments/2023-xue/gemm_res/lhr17.csr