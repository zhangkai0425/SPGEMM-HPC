#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <cuda.h>
#include <sys/time.h>
#include "common.h"
#include "utils.h"

extern const char* version_name;

int parse_args(int* reps, int p_id, int argc, char **argv);
int my_abort(int line, int code);
void main_gpufree(void *p);

#define MY_ABORT(ret) my_abort(__LINE__, ret)
#define ABORT_IF_ERROR(ret) CHECK_ERROR(ret, MY_ABORT(ret))
#define ABORT_IF_NULL(ret) CHECK_NULL(ret, MY_ABORT(NO_MEM))
#define INDENT "    "
#define TIME_DIFF(start, stop) 1.0 * (stop.tv_sec - start.tv_sec) + 1e-6 * (stop.tv_usec - start.tv_usec)

int main(int argc, char **argv) {
    int reps, i, ret;
    dist_matrix_t mat;
    dist_matrix_t matB;
    dist_matrix_t matC;
    double compute_time = 0, pre_time;
    struct timeval start, stop;

    ret = parse_args(&reps, 0, argc, argv);
    ABORT_IF_ERROR(ret)
    ret = read_matrix_default(&mat, argv[2]);
    ABORT_IF_ERROR(ret)
    ret = read_matrix_default(&matB, argv[2]);
    ABORT_IF_ERROR(ret)

    matC.additional_info = NULL;
    matC.CPU_free = free;
    matC.GPU_free = main_gpufree;

    printf("Benchmarking %s on %s.\n", version_name, argv[2]);
    printf(INDENT"%d x %d, %d non-zeros, %d run(s)\n", \
            mat.global_m, mat.global_m, mat.global_nnz, reps);
    printf(INDENT"Preprocessing.\n");
    
    gettimeofday(&start, NULL);
    preprocess(&mat, &matB);
    gettimeofday(&stop, NULL);
    pre_time = TIME_DIFF(start, stop);

    printf(INDENT"Testing.\n");

    gettimeofday(&start, NULL);
    spgemm(&mat, &matB, &matC);
    gettimeofday(&stop, NULL);
    compute_time = TIME_DIFF(start, stop);

    printf(INDENT"Checking.\n");
    ret = check_answer(&matC, argv[3]);
    // debug 
    if(ret == 0) {
        printf("\e[1;32m"INDENT"Result validated.\e[0m\n");
    } else {
        fprintf(stderr, "\e[1;31m"INDENT"Result NOT validated.\e[0m\n");
        MY_ABORT(ret);
    }
    destroy_dist_matrix(&mat);
    destroy_dist_matrix(&matB);
    destroy_dist_matrix(&matC);

    printf(INDENT INDENT"preprocess time = %lf s\n", pre_time);
    printf("\e[1;34m"INDENT INDENT"compute time = %lf s\e[0m\n", compute_time);

    return 0;
}

void main_gpufree(void *p) {
    cudaFree(p);
}


void print_help(const char *argv0, int p_id) {
    if(p_id == 0) {
        printf("\e[1;31mUSAGE: %s <repetitions> <input-file>\e[0m\n", argv0);
    }
}

int parse_args(int* reps, int p_id, int argc, char **argv) {
    int r;
    if(argc < 3) {
        print_help(argv[0], p_id);
        return 1;
    }
    r = atoi(argv[1]);
    if(r <= 0) {
        print_help(argv[0], p_id);
        return 1;
    }
    *reps = r;
    return SUCCESS;
}

int my_abort(int line, int code) {
    fprintf(stderr, "\e[1;33merror at line %d, error code = %d\e[0m\n", line, code);
    return fatal_error(code);
}

int fatal_error(int code) {
    exit(code);
    return code;
}
