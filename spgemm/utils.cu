#include <float.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cusparse.h>
#include "common.h"
#include "utils.h"

#define EPS_TOL 1e-4

#define CHECK_AND_SET(cond, state) if(cond) {state;}
#define CHECK_AND_BREAK(cond, state) if(cond) {state;break;}

typedef FILE *file_t;

int clean(int ret, void *p) {
    free(p);
    return ret;
}
                
int clean_file(int ret, file_t file) {
    fclose(file);
    return ret;
}

void gpu_free(void *p) {
    cudaFree(p);
}

int read_matrix_default(dist_matrix_t *mat, const char* filename) {
    file_t file;
    int global_m, global_nnz;
    int ret, count;
    index_t *r_pos;
    index_t *c_idx;
    data_t *values;
    index_t *gpu_r_pos;
    index_t *gpu_c_idx;
    data_t *gpu_values;

    file = fopen(filename, "rb");
    CHECK(file == NULL, IO_ERR)

    count = fread(&global_m, sizeof(index_t), 1, file);
    CHECK(count != 1, IO_ERR)

    r_pos = (index_t*)malloc(sizeof(index_t) * (global_m + 1));
    CHECK(r_pos == NULL, NO_MEM)

    count = fread(r_pos, sizeof(index_t), global_m + 1, file);
    CHECK(count != global_m + 1, IO_ERR)
    global_nnz = r_pos[global_m];

    c_idx = (index_t*)malloc(sizeof(index_t) * global_nnz);
    CHECK(c_idx == NULL, NO_MEM)
    values = (data_t*)malloc(sizeof(data_t) * global_nnz);
    CHECK(values == NULL, NO_MEM)

    count = fread(c_idx, sizeof(index_t), global_nnz, file);
    CHECK(count != global_nnz, IO_ERR)
    count = fread(values, sizeof(data_t), global_nnz, file);
    CHECK(count != global_nnz, IO_ERR)

    fclose(file);

    ret = cudaMalloc(&gpu_r_pos, sizeof(index_t) * (global_m + 1));
    CHECK_ERROR(ret, ret)
    ret = cudaMalloc(&gpu_c_idx, sizeof(index_t) * global_nnz);
    CHECK_ERROR(ret, ret)
    ret = cudaMalloc(&gpu_values, sizeof(data_t) * global_nnz);
    CHECK_ERROR(ret, ret)
    
    cudaMemcpy(gpu_r_pos, r_pos, sizeof(index_t) * (global_m + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_c_idx, c_idx, sizeof(index_t) * global_nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_values, values, sizeof(data_t)  * global_nnz, cudaMemcpyHostToDevice);

    mat->global_m = global_m;
    mat->global_nnz = global_nnz;
    mat->r_pos = r_pos;
    mat->c_idx = c_idx;
    mat->values = values;
    mat->gpu_r_pos = gpu_r_pos;
    mat->gpu_c_idx = gpu_c_idx;
    mat->gpu_values = gpu_values;
    mat->additional_info = NULL;
    mat->CPU_free = free;
    mat->GPU_free = gpu_free;
    return SUCCESS;
}

char *cancat_name(const char *a, const char *b) {
    int l1 = strlen(a) - 4, l2 = strlen(b);
    char *c = (char*)malloc(sizeof(char) * (l1 + l2 + 1));
    if(c != NULL) {
        memcpy(c, a, l1 * sizeof(char));
        memcpy(c + l1, b, l2 * sizeof(char));
        c[l1 + l2] = '\0';
    }
    return c;
}


int read_vector(dist_matrix_t *mat, const char* filename, const char* suffix, int n, data_t* x) {
    char *new_name;
    file_t file;
    int count, i, m = mat->global_m;
    new_name = cancat_name(filename, suffix);
    CHECK(new_name == NULL, NO_MEM)

    file = fopen(new_name, "rb");
    CHECK(new_name == NULL, clean(IO_ERR, new_name))

    for(i = 0; i < n; ++i) {
        count = fread(x + m * i, sizeof(data_t), m, file);
        CHECK(count != m, clean(clean_file(IO_ERR, file), new_name))
    }
    return clean(clean_file(SUCCESS, file), new_name);
}

int check_answer(dist_matrix_t *mat, const char* filename) {
    FILE *file;
    int res_m, res_nnz;
    int *res_r_pos, *res_c_idx;
    float *res_values;
    int i,j;
    int x;

    file = fopen(filename, "rb");
    fread(&res_m, sizeof(index_t), 1, file);
    res_r_pos = (index_t*)malloc(sizeof(index_t) * (res_m + 1));
    fread(res_r_pos, sizeof(index_t), res_m + 1, file);
    res_nnz = res_r_pos[res_m];
    res_c_idx = (index_t*)malloc(sizeof(index_t) * res_nnz);
    res_values = (data_t*)malloc(sizeof(data_t) * res_nnz);
    fread(res_c_idx, sizeof(index_t), res_nnz, file);
    fread(res_values, sizeof(data_t), res_nnz, file);
    fclose(file);

    int *resid, *pos;
    resid = (int*) malloc(sizeof(int)*(res_m+1));
    pos = (int*) malloc(sizeof(int)*(res_m+1));
    int resi_num;
    float *resi_val;
    float *std_resi;
    float *res_row, *res_col;
    float std;
    res_row = (float*)malloc(sizeof(float)*(res_m+1));
    res_col = (float*)malloc(sizeof(float)*(res_m+1));
    resi_val = (float*)malloc(sizeof(float)*(res_m+1));
    std_resi = (float*)malloc(sizeof(float)*(res_m+1));

    for (i = 0; i < res_m; i++){
        res_row[i] = 0;
        res_col[i] = 0;
        pos[i] = -1;
    }
    for (i = 0; i < res_m; i++) {
        for (j = mat->r_pos[i]; j < mat->r_pos[i+1]; j++) {
            x = mat->c_idx[j];
            res_row[i] = res_row[i] + (mat->values[j]) * (mat->values[j]);
            res_col[x] = res_col[x] + (mat->values[j]) * (mat->values[j]);
        }
    }
    for (i = 0; i < res_m; i++) {
        res_row[i] = sqrt(res_row[i]);
        res_col[i] = sqrt(res_col[i]);
    }

    for (i = 0; i < res_m; i++) {
        resi_num = 0;
        for (j = mat->r_pos[i]; j < mat->r_pos[i+1]; j++) {
            x = mat->c_idx[j];
            if (pos[x] != i) {
                pos[x] = i;
                resid[resi_num] = x;
                resi_num++;
                std_resi[x] = 0;
                resi_val[x] = 0; 
            }
            resi_val[x] = resi_val[x] + mat->values[j];
            std_resi[x] = mat->values[j];
        }
        for (j = res_r_pos[i]; j < res_r_pos[i+1]; j++) {
            x = res_c_idx[j];
            if (pos[x] != i) {
                pos[x] = i;
                resid[resi_num] = x;
                resi_num++;
                resi_val[x] = 0; 
                std_resi[x] = 0;
            }
            resi_val[x] = resi_val[x] - res_values[j];
            if (std_resi[x] == 0) std_resi[x] = res_values[j];
        }
        for (j = 0; j < resi_num; j++) {
            x = resid[j];
            if (resi_val[x] < 0) resi_val[x] = -resi_val[x];
            if (std_resi[x] < 0) std_resi[x] = -std_resi[x];
            std = res_row[i] * res_col[x] * 1e-3 + 1e-6;
            if (resi_val[x] > std) return -1;
        }
    }
    return 0;
}

void destroy_dist_matrix(dist_matrix_t *mat) {
    if(mat->additional_info != NULL){
        destroy_additional_info(mat->additional_info);
        mat->additional_info = NULL;
    }
    if(mat->CPU_free != NULL) {
        if(mat->r_pos != NULL){
            mat->CPU_free(mat->r_pos);
            mat->r_pos = NULL;
        }
        if(mat->c_idx != NULL){
            mat->CPU_free(mat->c_idx);
            mat->c_idx = NULL;
        }
        if(mat->values != NULL){
            mat->CPU_free(mat->values);
            mat->values = NULL;
        }
    }
    if(mat->GPU_free != NULL) {
        if(mat->gpu_r_pos != NULL){
            mat->GPU_free(mat->gpu_r_pos);
            mat->gpu_r_pos = NULL;
        }
        if(mat->gpu_c_idx != NULL){
            mat->GPU_free(mat->gpu_c_idx);
            mat->gpu_c_idx = NULL;
        }
        if(mat->gpu_values != NULL){
            mat->GPU_free(mat->gpu_values);
            mat->gpu_values = NULL;
        }
    }
}
