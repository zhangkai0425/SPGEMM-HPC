#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "common.h"
#include "utils.h"

const char* version_name = "optimized version 0 : naive implementation";\

typedef struct {
    int est_c_nnz = 0;
} additional_info_t;

typedef additional_info_t* info_ptr_t;

void preprocess(dist_matrix_t *matA, dist_matrix_t *matB) {
    info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
    // Estimate nnz of C
    int est_c_nnz = 0;
    std::vector<int> mask(matA->global_m, -1);
    for (int i = 0; i < matA->global_m; ++i) {
        int row_nnz = 0;
        int row_begin = matA->r_pos[i];
        int row_end = matA->r_pos[i + 1];
        for (int j = row_begin; j < row_end; ++j) {
            int col_idx = matA->c_idx[j];
            for (int k = matB->r_pos[col_idx]; k < matB->r_pos[col_idx + 1]; ++k) {
                int col_idx_b = matB->c_idx[k];
                if (mask[col_idx_b] != i)
                {
                    mask[col_idx_b] = i;
                    row_nnz ++;
                }
            }
        }
        est_c_nnz += row_nnz;
    }
    p->est_c_nnz = est_c_nnz;
    matA->additional_info = p;
    printf("estimate c nnz: %d \n", est_c_nnz);
}

void destroy_additional_info(void *additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    free(p);
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

void spgemm(dist_matrix_t *matA, dist_matrix_t *matB, dist_matrix_t *matC) {
    info_ptr_t p = (info_ptr_t)matA->additional_info;
    int est_c_nnz = p->est_c_nnz;
    int c_nnz = 0;
    //please put your result back in matC->r_pos/c_idx/values in CSR format
    matC->global_m = matA->global_m;
    matC->r_pos = (index_t*)malloc(sizeof(int)*(matC->global_m+1));
    matC->c_idx = (index_t*)malloc(sizeof(int) * (est_c_nnz));
    matC->values = (data_t*)malloc(sizeof(int) * (est_c_nnz));
    std::vector<data_t> c_row(matC->global_m,0);
    matC->r_pos[0] = 0;
    
    // C = A * B using CSR format
    for (int i = 0; i < matA->global_m; ++i) {
        int row_start_A = matA->r_pos[i];
        int row_end_A = matA->r_pos[i + 1];
        // A[i][j]
        for (int row_A = row_start_A; row_A < row_end_A; ++row_A) {
            int col_A = matA->c_idx[row_A]; 
            float val_A = matA->values[row_A]; 

            int row_start_B = matB->r_pos[col_A];
            int row_end_B = matB->r_pos[col_A + 1]; 
            // C[i][:] += A[i][j] * B[j][:]
            for (int row_B = row_start_B; row_B < row_end_B; ++row_B) {
                int col_B = matB->c_idx[row_B];
                c_row[col_B] += val_A * matB->values[row_B];
            }
        }
        for (index_t idx = 0; idx < matC->global_m; idx++) {
            if(c_row[idx] != 0.0f) {
                matC->c_idx[c_nnz] = idx;
                matC->values[c_nnz] = c_row[idx];
                c_nnz ++;
                c_row[idx] = 0;
            }
        }
        matC->r_pos[i + 1] = c_nnz;
    }
    matC->global_nnz = c_nnz;
    printf("Result report , global_nnz = %d\n", matC->global_nnz);

    // for (int i = 0; i < 100; i++)
    //     printf("%f\n", matC->values[i]);
    FILE* file = fopen("real.txt", "w");

    // 检查文件是否成功打开
    if (file == NULL) {
        fprintf(stderr, "无法打开文件 fake.txt\n");
        return;
    }

    // 循环写入数据到文件
    for (int i = 0; i < matC->global_m; i++) {
        fprintf(file, "%d\n", matC->r_pos[i + 1] - matC->r_pos[i]);
    }

    // 关闭文件
    fclose(file);
}
