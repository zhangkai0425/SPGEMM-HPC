#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <vector>
#include "common.h"
#include "utils.h"

const char* version_name = "optimized version 2 : dividing by rows and cols";\

#define CheckCudaErrors(ret) if(ret != cudaSuccess) { fprintf(stderr, "error %d in line %d\n", ret, __LINE__);}

#define CUDA_CHECK_ERROR(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error encountered at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::cerr << "Error code: " << err << ", " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define WARP_SIZE 32

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

typedef struct {
    int est_c_nnz = 0;
    dim3 block;
    dim3 grid;
    int* est_c_nnz_rows_begin;
    int* est_c_nnz_rows_end;
} additional_info_t;

// Hash Table
struct HashNode {
    index_t col_idx;
    data_t val;
};

typedef additional_info_t* info_ptr_t;

void preprocess(dist_matrix_t *matA, dist_matrix_t *matB) {
    info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));

    // Estimate nnz of C
    int est_c_nnz = 0;
    int* est_c_nnz_rows_begin = (int*)malloc(sizeof(int) * matA->global_m);
    int* est_c_nnz_rows_end = (int*)malloc(sizeof(int) * matA->global_m);

    std::vector<int> mask(matA->global_m, -1);
    for (int i = 0; i < matA->global_m; ++i) {
        int row_nnz = 0;
        int row_begin = matA->r_pos[i];
        int row_end = matA->r_pos[i + 1];
        for (int j = row_begin; j < row_end; ++j) {
            int c_idx = matA->c_idx[j];
            for (int k = matB->r_pos[c_idx]; k < matB->r_pos[c_idx + 1]; ++k) {
                int c_idx_b = matB->c_idx[k];
                if (mask[c_idx_b] != i) {
                    mask[c_idx_b] = i;
                    row_nnz++;
                }
            }
        }
        est_c_nnz_rows_begin[i] = est_c_nnz;
        est_c_nnz += row_nnz;
        est_c_nnz_rows_end[i] = est_c_nnz;
    }
    p->est_c_nnz = est_c_nnz;
    p->est_c_nnz_rows_begin = est_c_nnz_rows_begin;
    p->est_c_nnz_rows_end = est_c_nnz_rows_end;
    matA->additional_info = p;
    printf("Estimate c nnz succeeded ! est_c_nnz : %d \n", est_c_nnz);

    dim3 block,grid;
    block.x = 1;
    block.y = WARP_SIZE;
    grid.x = matA->global_m;
    grid.y = ceiling(matA->global_m,WARP_SIZE);
    p->block = block;
    p->grid = grid;
    printf("Threads block cutting succeeded ! \n");
}

void destroy_additional_info(void *additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    free(p);
}

__global__ void initializeHashTable(HashNode* d_HashTable, int tableSize) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < tableSize)
        d_HashTable[tid].col_idx = -1;
}

__device__ int mutex = 0;

__global__ void spgemm_kernel  (const index_t* d_r_pos_A,const index_t* d_c_idx_A,const data_t* d_val_A,
                                const index_t* d_r_pos_B,const index_t* d_c_idx_B,const data_t* d_val_B,
                                index_t* d_r_pos_C,index_t* d_c_idx_C,data_t* d_val_C,
                                const index_t* d_est_c_nnz_rows_begin, const index_t* d_est_c_nnz_rows_end, 
                                HashNode* d_HashTable, 
                                const int global_m) {
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    int col_id_idx = blockIdx.y * WARP_SIZE + threadIdx.y;
    
    if (row_id >= global_m) return;

    // C = A * B using CSR format
    int ptr_a_begin = d_r_pos_A[row_id];
    int ptr_a_end = d_r_pos_A[row_id + 1];

    // Shared data of row A
    __shared__ int sm_idx_A[WARP_SIZE];
    __shared__ float sm_val_A[WARP_SIZE];

    int tableSize = d_est_c_nnz_rows_end[row_id] - d_est_c_nnz_rows_begin[row_id];
    int hash_begin = d_est_c_nnz_rows_begin[row_id];
    HashNode* Row_HashTable = d_HashTable + hash_begin;

    // A[i][j]
    for (int ptr_a = ptr_a_begin; ptr_a < ptr_a_end; ptr_a += WARP_SIZE) {
        int thr_ptr_a = ptr_a + threadIdx.y;
        if (thr_ptr_a < ptr_a_end) {
            sm_idx_A[threadIdx.y] = d_c_idx_A[thr_ptr_a];
            sm_val_A[threadIdx.y] = d_val_A[thr_ptr_a];
        }
        __syncthreads();
        int sm_end = min(WARP_SIZE, ptr_a_end - ptr_a);
        // C[i][:] += A[i][j] * B[j][:]
        for (int i = 0; i < sm_end; ++i) {
            index_t col_A = sm_idx_A[i];
            index_t ptr_b_begin = d_r_pos_B[col_A];
            index_t ptr_b_end = d_r_pos_B[col_A + 1];
            index_t nnz_b_row = ptr_b_end - ptr_b_begin;
            if (col_id_idx < nnz_b_row) {
                index_t col_B = d_c_idx_B[ptr_b_begin + col_id_idx];
                data_t val = sm_val_A[i] * d_val_B[ptr_b_begin + col_id_idx];
                // Calculate hash value
                int hash = col_B % tableSize;
                // Traverse the nodes in a hash table, using linear probing to handle collisions
                int index = hash;
                bool found = false;
                while (!found) {
                    // Use atomic operations to read information from hash table nodes.
                    int stored_col_idx = atomicAdd(&(Row_HashTable[index].col_idx), 0);
                    if (stored_col_idx == -1) {
                        // If a matching col_idx is found, update the value.
                        int old_val = atomicCAS(&(Row_HashTable[index].col_idx), -1, col_B);
                        if (old_val == -1) {
                            // Successfully claimed the slot, now write the value
                            atomicExch(&(Row_HashTable[index].val), val);
                            found = true;
                        }
                    } else if (stored_col_idx == col_B) {
                        // Found matching col_idx, update the value
                        atomicAdd(&(Row_HashTable[index].val), val);
                        found = true;
                    }
                    // Move to the next slot using linear probing
                    index = (index + 1) % tableSize;
                }
            }
        }
    }
}

__global__ void spgemm_kernel_finalize(index_t* d_r_pos_C,
                                       index_t* d_c_idx_C,
                                       data_t* d_val_C,
                                       HashNode* d_HashTable,
                                       const index_t* d_est_c_nnz_rows_begin,
                                       const index_t* d_est_c_nnz_rows_end) {
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    int hash_begin = d_est_c_nnz_rows_begin[row_id];
    HashNode* Row_HashTable = d_HashTable + hash_begin;
    if (threadIdx.y == 0) {
        int col_cnt = 0;
        int col_idx = 0;
        int table_begin = d_est_c_nnz_rows_begin[row_id];
        int table_end = d_est_c_nnz_rows_end[row_id];
        for (int j = 0; j < table_end - table_begin; j++) {
            if (Row_HashTable[j].val != 0.0f) {
                col_idx = Row_HashTable[j].col_idx;
                d_c_idx_C[table_begin + col_cnt] = col_idx;
                d_val_C[table_begin + col_cnt] = Row_HashTable[j].val;
                col_cnt++;
            }
        }
        d_r_pos_C[row_id] = col_cnt;
    }
}

// Kernel to count non-zero elements in each row
__global__ void CountNonZeroElements(const index_t* d_r_pos_C,
                                     index_t* m_r_pos_C,
                                     const int global_m) {
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id == 0) {
        int c_nnz = 0;
        m_r_pos_C[0] = 0;
        for (int i = 0; i < global_m; ++i) {
            c_nnz += d_r_pos_C[i];
            m_r_pos_C[i + 1] = c_nnz;
        }
    }
}

// Kernel to count non-zero elements in each row
__global__ void ToRealCSR (const index_t* d_r_pos_C, const index_t* d_c_idx_C, const data_t* d_val_C, 
                            index_t* m_r_pos_C, index_t* m_c_idx_C, data_t* m_val_C,
                            const index_t* d_est_c_nnz_rows_begin, const int global_m) {
    int row_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_id >= global_m) return;
    int idx_begin_origin = d_est_c_nnz_rows_begin[row_id];
    int idx_begin_new = m_r_pos_C[row_id];
    for (int j=0; j<d_r_pos_C[row_id]; ++j) {
        m_c_idx_C[idx_begin_new + j] = d_c_idx_C[idx_begin_origin + j];
        m_val_C[idx_begin_new + j] = d_val_C[idx_begin_origin + j];
    }
}

void spgemm(dist_matrix_t *matA, dist_matrix_t *matB, dist_matrix_t *matC) {
    
    info_ptr_t p = (info_ptr_t)matA->additional_info;
    int est_c_nnz = p->est_c_nnz;
    int global_m = matA->global_m;

    index_t* est_c_nnz_rows_begin = p->est_c_nnz_rows_begin;
    index_t* est_c_nnz_rows_end = p->est_c_nnz_rows_end;

    // Put result back in matC->r_pos/c_idx/values in CSR format
    matC->global_m = global_m;
    matC->r_pos = (int*)malloc(sizeof(int) * (global_m + 1));
    matC->c_idx = (int*)malloc(sizeof(int) * (est_c_nnz));
    matC->values = (data_t*)malloc(sizeof(int) * (est_c_nnz));

    // Allocate device memory
    index_t* d_r_pos_C;
    index_t* d_c_idx_C;
    data_t* d_val_C;

    index_t* d_est_c_nnz_rows_begin;
    index_t* d_est_c_nnz_rows_end;

    int tableSize = est_c_nnz;
    HashNode* d_HashTable;
    cudaMalloc((void**)&d_HashTable, tableSize * sizeof(HashNode));

    CUDA_CHECK_ERROR(cudaMalloc(&d_est_c_nnz_rows_begin, sizeof(index_t) * global_m));
    CUDA_CHECK_ERROR(cudaMalloc(&d_est_c_nnz_rows_end, sizeof(index_t) * global_m));
    CUDA_CHECK_ERROR(cudaMalloc(&d_r_pos_C, sizeof(index_t) * (global_m + 1)));
    CUDA_CHECK_ERROR(cudaMalloc(&d_c_idx_C, sizeof(index_t) * est_c_nnz));
    CUDA_CHECK_ERROR(cudaMalloc(&d_val_C, sizeof(data_t) * est_c_nnz));

    CUDA_CHECK_ERROR(cudaMalloc(&matC->gpu_r_pos, sizeof(index_t) * (global_m + 1)));
    CUDA_CHECK_ERROR(cudaMalloc(&matC->gpu_c_idx, sizeof(index_t) * est_c_nnz));
    CUDA_CHECK_ERROR(cudaMalloc(&matC->gpu_values, sizeof(data_t) * est_c_nnz));

    CUDA_CHECK_ERROR(cudaMemcpy(d_est_c_nnz_rows_begin, est_c_nnz_rows_begin, sizeof(index_t) * global_m, cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_est_c_nnz_rows_end, est_c_nnz_rows_end, sizeof(index_t) * global_m, cudaMemcpyHostToDevice));

    // Initialize hash table
    int initBlockSize = 256;
    int initGridSize = ceiling(tableSize,initBlockSize);
    initializeHashTable<<<initGridSize, initBlockSize>>>(d_HashTable, tableSize);
    cudaDeviceSynchronize(); 

    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
        printf("CUDA error after kernel initializeHashTable: %s\n", cudaGetErrorString(cuda_err));

    // Spgemm kernel
    dim3 dim_Block = p->block;
    dim3 dim_Grid = p->grid;

    spgemm_kernel<<<dim_Grid, dim_Block>>> (matA->gpu_r_pos,matA->gpu_c_idx,matA->gpu_values,
                                            matB->gpu_r_pos,matB->gpu_c_idx,matB->gpu_values,
                                            d_r_pos_C,d_c_idx_C,d_val_C,
                                            d_est_c_nnz_rows_begin,d_est_c_nnz_rows_end,
                                            d_HashTable,
                                            global_m);

    cudaDeviceSynchronize();

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
        printf("CUDA error after kernel spgemm_kernel: %s\n", cudaGetErrorString(cuda_err));

    spgemm_kernel_finalize<<<dim_Grid, dim_Block>>>(d_r_pos_C,d_c_idx_C,d_val_C,d_HashTable,
                                                    d_est_c_nnz_rows_begin,d_est_c_nnz_rows_end);

    cudaDeviceSynchronize();
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
        printf("CUDA error after kernel spgemm_kernel_finalize: %s\n", cudaGetErrorString(cuda_err));                       

    // To real CSR format
    
    int trans_block = 1;
    int trans_grid = 1;

    CountNonZeroElements<<<trans_grid, trans_block>>>(d_r_pos_C, matC->gpu_r_pos, global_m);

    trans_block = 256;
    trans_grid = ceiling(global_m, trans_block);
    
    ToRealCSR<<<trans_grid, trans_block>>> (d_r_pos_C, d_c_idx_C, d_val_C, 
                                            matC->gpu_r_pos, matC->gpu_c_idx, matC->gpu_values,
                                            d_est_c_nnz_rows_begin, global_m);
    cudaDeviceSynchronize();

    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess)
        printf("CUDA error after kernel ToRealCSR: %s\n",
               cudaGetErrorString(cuda_err));

    CUDA_CHECK_ERROR(cudaMemcpy(matC->r_pos, matC->gpu_r_pos, sizeof(index_t) * (global_m + 1), cudaMemcpyDeviceToHost));
    matC->r_pos[0] = 0;
    matC->global_nnz = matC->r_pos[global_m];

    printf("Result report , global_nnz = %d\n", matC->global_nnz);

    CUDA_CHECK_ERROR(cudaMemcpy(matC->c_idx, matC->gpu_c_idx, sizeof(index_t) * matC->global_nnz, cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(matC->values, matC->gpu_values, sizeof(data_t) * matC->global_nnz, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK_ERROR(cudaFree(d_r_pos_C));
    CUDA_CHECK_ERROR(cudaFree(d_c_idx_C));
    CUDA_CHECK_ERROR(cudaFree(d_val_C));
    CUDA_CHECK_ERROR(cudaFree(d_est_c_nnz_rows_begin));
    CUDA_CHECK_ERROR(cudaFree(d_est_c_nnz_rows_end));
}