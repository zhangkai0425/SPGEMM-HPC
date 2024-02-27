#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include <cusparse.h>
#include "common.h"

const char* version_name = "cuSPARSE SpMV";\

#define CHECK_CUSPARSE(ret) if(ret != CUSPARSE_STATUS_SUCCESS) { fprintf(stderr, "error %d in line %d\n", ret, __LINE__);}

typedef struct {
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA, matB, matC;
    cusparseSpGEMMDescr_t gemmDesc;
    void *dBuffer1, *dBuffer2;
    size_t bufferSize1, bufferSize2;

} additional_info_t;

typedef additional_info_t *info_ptr_t;

void preprocess(dist_matrix_t *matA, dist_matrix_t *matB) {
    info_ptr_t p = (info_ptr_t)malloc(sizeof(additional_info_t));
    cusparseCreate(&p->handle);
    cusparseCreateCsr(&p->matA, matA->global_m, matA->global_m, matA->global_nnz, matA->gpu_r_pos, matA->gpu_c_idx, matA->gpu_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&p->matB, matB->global_m, matB->global_m, matB->global_nnz, matB->gpu_r_pos, matB->gpu_c_idx, matB->gpu_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateCsr(&p->matC, matA->global_m, matB->global_m, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    
    cusparseSpGEMM_createDescr(&p->gemmDesc);


    //cusparseSetMatIndexBase(p->descrA, CUSPARSE_INDEX_BASE_ZERO);
    //cusparseSetMatType(p->descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    matA->additional_info = p;
}

void destroy_additional_info(void *additional_info) {
    info_ptr_t p = (info_ptr_t)additional_info;
    cusparseSpGEMM_destroyDescr(p->gemmDesc);
    cusparseDestroySpMat(p->matA);
    cusparseDestroySpMat(p->matB);
    cusparseDestroySpMat(p->matC);
    cusparseDestroy(p->handle);
    cudaFree(p->dBuffer1);
    cudaFree(p->dBuffer2);
    free(p);
}

void spgemm(dist_matrix_t *mat, dist_matrix_t *matB, dist_matrix_t *matC) {
    int m = mat->global_m, nnz = mat->global_nnz;
    const data_t alpha = 1.0, beta = 0.0;
    info_ptr_t p = (info_ptr_t)mat->additional_info;

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p->matA, p->matB, &beta, 
        p->matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, p->gemmDesc, &p->bufferSize1, NULL))
    p->dBuffer1 = NULL;
    cudaMalloc( &p->dBuffer1, p->bufferSize1);
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p->matA, p->matB, &beta, 
        p->matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, p->gemmDesc, &p->bufferSize1, p->dBuffer1))

    CHECK_CUSPARSE (cusparseSpGEMM_compute(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p->matA, p->matB, &beta, 
        p->matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, p->gemmDesc, &p->bufferSize2, NULL))
    p->dBuffer2 = NULL;
    cudaMalloc( &p->dBuffer2, p->bufferSize2);
    cusparseSpGEMM_compute(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p->matA, p->matB, &beta, 
        p->matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, p->gemmDesc, &p->bufferSize2, p->dBuffer2);

    int64_t C_row, C_col, C_nnz;
    cusparseSpMatGetSize(p->matC, &C_row, &C_col, &C_nnz);
    matC->global_m = C_row;
    matC->global_nnz = C_nnz;
    cudaMalloc(&matC->gpu_r_pos, (C_row+1)*sizeof(int));
    cudaMalloc(&matC->gpu_c_idx, C_nnz*sizeof(int));
    cudaMalloc(&matC->gpu_values, C_nnz*sizeof(float));
    matC->r_pos = (int*)malloc(sizeof(int)*(C_row+1));
    matC->c_idx = (int*)malloc(sizeof(int)*C_nnz);
    matC->values = (float*)malloc(sizeof(float)*C_nnz);
    printf("Real nnz = %d \n",C_nnz);

    cusparseCsrSetPointers(p->matC, matC->gpu_r_pos, matC->gpu_c_idx, matC->gpu_values);
    cusparseSpGEMM_copy(p->handle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
        CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, p->matA, p->matB, &beta, 
        p->matC, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, p->gemmDesc);
        
    cudaMemcpy(matC->r_pos, matC->gpu_r_pos, (C_row+1)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matC->c_idx, matC->gpu_c_idx, (C_nnz)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(matC->values, matC->gpu_values, (C_nnz)*sizeof(float), cudaMemcpyDeviceToHost);

}
