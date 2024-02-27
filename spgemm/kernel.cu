#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

// 第一个核函数，计算 d_dense_C 存储到方阵 C 的行
__global__ void sparseMatrixMultiplication(const int global_m,
                                           const int* d_r_pos_A,const int* d_c_idx_A,const float* d_val_A,
                                           const int* row_ptr_B,const int* d_c_idx_B,const float* d_val_B,
                                           float* d_dense_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < global_m) {
        int row_start_A = d_r_pos_A[i];
        int row_end_A = d_r_pos_A[i + 1];

        for (int row_A = row_start_A; row_A < row_end_A; ++row_A) {
            int col_A = d_c_idx_A[row_A];
            float val_A = d_val_A[row_A];

            int row_start_B = row_ptr_B[col_A];
            int row_end_B = row_ptr_B[col_A + 1];

            for (int row_B = row_start_B + threadIdx.y; row_B < row_end_B; row_B += blockDim.y) {
                int col_B = d_c_idx_B[row_B];
                atomicAdd(&d_dense_C[i * global_m + col_B],val_A * d_val_B[row_B]);
            }
            __syncthreads();
        }
    }
}

// 第二个核函数，将方阵 C 转化为 CSR 格式
__global__ void denseToCSR(const int global_m,const float* d_dense_C,int* d_c_idx_C,float* d_val_C,int* d_r_pos_C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < global_m) {
        int row_start = i * global_m;
        int row_end = (i + 1) * global_m;

        int nonzero_count = 0;
        for (int j = row_start; j < row_end; ++j) {
            if (d_dense_C[j] != 0.0f) {
                d_c_idx_C[nonzero_count] = j % global_m;
                d_val_C[nonzero_count] = d_dense_C[j];
                ++nonzero_count;
            }
        }
        d_r_pos_C[i + 1] = nonzero_count;
    }
}

// Good,but I don't know what to do next,cause there's so many things for me to consider !
// Oh! That's really hard! 
// I think there's two important things for you to do:
// 1. You should check the grid size to cover all data.
// 2. You should understand the X Y thread block and use it well! I believe you are not truly understand the 2-D thread block yet.
e