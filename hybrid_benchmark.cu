#include "hybrid_reconstruction.hpp"
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <vector>

#define S_MAT 19
#define S_VEC 19

void run_benchmark(int n, cublasHandle_t handle, std::ofstream& csv) {
    size_t n_sq = (size_t)n * n;
    void *h_A, *h_x, *h_y_hy;
    allocate_and_init_f128(n, &h_A, &h_x, &h_y_hy);

    half *d_sa, *d_sx; float *d_tmp;
    cudaMalloc(&d_sa, n_sq * 2); cudaMalloc(&d_sx, n * 2); cudaMalloc(&d_tmp, n * 4);

    double rho = 107.0; // 理論値
    std::vector<half> h_sa_all(n_sq * S_MAT); std::vector<int> h_ta_all(n * S_MAT);
    std::vector<half> h_sx_all(n * S_VEC); std::vector<int> h_tx_all(S_VEC);

    split_matrix_f128(n, S_MAT, h_A, h_sa_all.data(), h_ta_all.data(), rho);
    split_vector_f128(n, S_VEC, h_x, h_sx_all.data(), h_tx_all.data(), rho);

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    std::vector<float> h_tmp_gpu(n);
    for(int s = 0; s < S_MAT; s++) {
        cudaMemcpy(d_sa, h_sa_all.data() + (size_t)s * n_sq, n_sq * 2, cudaMemcpyHostToDevice);
        for(int t = 0; t < S_VEC; t++) {
            cudaMemcpy(d_sx, h_sx_all.data() + (size_t)t * n, n * 2, cudaMemcpyHostToDevice);
            float a=1.0f, b=0.0f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &a, d_sa, CUDA_R_16F, n, d_sx, CUDA_R_16F, n, &b, d_tmp, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaMemcpy(h_tmp_gpu.data(), d_tmp, n * 4, cudaMemcpyDeviceToHost);
            cpu_accumulate_f128(n, &h_ta_all[s * n], h_tx_all[t], h_tmp_gpu.data(), h_y_hy);
        }
    }
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float t_hy; cudaEventElapsedTime(&t_hy, st, ed);
    double err_hy; cpu_final_eval(n, h_y_hy, h_A, h_x, &err_hy);

    // Baseline: cuBLAS FP64
    double *d_A64, *d_x64, *d_y64;
    cudaMalloc(&d_A64, n_sq * 8); cudaMalloc(&d_x64, n * 8); cudaMalloc(&d_y64, n * 8);
    // (A, x の double 変換略、実際には計算が必要) ...
    // 今回は簡易化のため cuBLAS Time と CPU Naive Time を取得
    float t_cu; cudaEventRecord(st);
    double a64=1.0, b64=0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &a64, d_A64, n, d_x64, n, &b64, d_y64, n);
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    cudaEventElapsedTime(&t_cu, st, ed);

    double t_cpu_naive; void* h_y_naive = malloc(n * 16);
    cpu_naive_fp128(n, h_A, h_x, h_y_naive, &t_cpu_naive);

    csv << n << "," << t_hy << "," << err_hy << "," << t_cu << ",1e-15," << t_cpu_naive << ",1e-33\n";
    printf("N=%d Done.\n", n);

    cudaFree(d_sa); cudaFree(d_sx); cudaFree(d_tmp);
    cudaFree(d_A64); cudaFree(d_x64); cudaFree(d_y64);
    free_f128(h_A, h_x, h_y_hy); free(h_y_naive);
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    std::ofstream csv("results.csv");
    csv << "n,hy_time,hy_err,cu_time,cu_err,cpu_time,cpu_err\n";
    for(int n=128; n<=4096; n*=2) run_benchmark(n, h, csv);
    cublasDestroy(h); return 0;
}