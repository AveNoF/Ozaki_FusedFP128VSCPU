#include "hybrid_reconstruction.hpp"
#include <cublas_v2.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>

#define S_MAT 19
#define S_VEC 19

void run_benchmark(int n, cublasHandle_t handle, std::ofstream& csv) {
    size_t n_sq = (size_t)n * n;
    void *h_A, *h_x, *h_y_hy, *y_truth;
    allocate_and_init_f128(n, &h_A, &h_x, &h_y_hy);
    y_truth = malloc(n * 16);

    printf("N=%-5d | Truth FP256 Computation...\n", n);
    cpu_calc_truth_fp256(n, h_A, h_x, y_truth);

    half *d_sa, *d_sx; float *d_tmp;
    cudaMalloc(&d_sa, (size_t)n * n * 2); cudaMalloc(&d_sx, n * 2); cudaMalloc(&d_tmp, n * 4);
    
    // 理論に基づき rho を動的に計算 
    // rho = ceil(-log2(u_target) + (log2(u_hw) + log2(n))/2)
    // Target: FP128 (113bit), HW: FP32 Accum (24bit)
    double rho = ceil(113.0 + (-24.0 + log2(n)) / 2.0);
    
    std::vector<half> h_sa_all((size_t)n * n * S_MAT); std::vector<int> h_ta_all(n * S_MAT);
    std::vector<half> h_sx_all(n * S_VEC); std::vector<int> h_tx_all(S_VEC);
    split_matrix_f128(n, S_MAT, h_A, h_sa_all.data(), h_ta_all.data(), rho);
    split_vector_f128(n, S_VEC, h_x, h_sx_all.data(), h_tx_all.data(), rho);

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    std::vector<float> h_tmp_gpu(n);
    for(int s = 0; s < S_MAT; s++) {
        cudaMemcpy(d_sa, h_sa_all.data() + (size_t)s*n_sq, (size_t)n*n*2, cudaMemcpyHostToDevice);
        for(int t = 0; t < S_VEC; t++) {
            cudaMemcpy(d_sx, h_sx_all.data() + (size_t)t*n, n*2, cudaMemcpyHostToDevice);
            float a=1.0f, b=0.0f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &a, d_sa, CUDA_R_16F, n, d_sx, CUDA_R_16F, n, &b, d_tmp, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaMemcpy(h_tmp_gpu.data(), d_tmp, n * 4, cudaMemcpyDeviceToHost);
            cpu_accumulate_f128(n, &h_ta_all[s * n], h_tx_all[t], h_tmp_gpu.data(), h_y_hy);
        }
    }
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float t_hy; cudaEventElapsedTime(&t_hy, st, ed);
    double err_hy = measure_rmse_mpfr(n, h_A, h_x, h_y_hy);

    double *d_A64, *d_x64, *d_y64;
    cudaMalloc(&d_A64, (size_t)n * n * 8); cudaMalloc(&d_x64, n * 8); cudaMalloc(&d_y64, n * 8);
    cudaEventRecord(st);
    double a64=1.0, b64=0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &a64, d_A64, n, d_x64, n, &b64, d_y64, n);
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float t_cu64; cudaEventElapsedTime(&t_cu64, st, ed);

    double t_cpu128, err_cpu128, t_cpu64, err_cpu64;
    cpu_benchmark_baselines(n, h_A, h_x, y_truth, &t_cpu128, &err_cpu128, &t_cpu64, &err_cpu64);

    csv << n << "," << t_hy << "," << std::scientific << std::setprecision(25) << err_hy << "," 
        << t_cu64 << "," << t_cpu128 << "," << err_cpu128 << "," << t_cpu64 << "," << err_cpu64 << "\n";
    printf("N=%-5d | RMSE: Ozaki=%.1e, CPU128=%.1e | rho=%d\n", n, err_hy, err_cpu128, (int)rho);

    cudaFree(d_sa); cudaFree(d_sx); cudaFree(d_tmp);
    cudaFree(d_A64); cudaFree(d_x64); cudaFree(d_y64);
    free_f128(h_A, h_x, h_y_hy); free(y_truth);
}

int main(int argc, char** argv) {
    int n_min = 128, n_max = 4096, n_step = 2;
    if(argc >= 2) n_min = atoi(argv[1]);
    if(argc >= 3) n_max = atoi(argv[2]);
    if(argc >= 4) n_step = atoi(argv[3]);

    cublasHandle_t h; cublasCreate(&h);
    std::ofstream csv("results.csv");
    csv << "n,hy_time,hy_err,cu64_time,cpu128_time,cpu128_err,cpu64_time,cpu64_err\n";
    for(int n = n_min; n <= n_max; n *= n_step) run_benchmark(n, h, csv);
    csv.close(); cublasDestroy(h); return 0;
}