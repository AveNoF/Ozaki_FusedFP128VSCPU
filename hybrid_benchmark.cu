#include "hybrid_reconstruction.hpp"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

#define S_MAT 16
#define S_VEC 16

void run_safe_test(int n, cublasHandle_t handle) {
    size_t n_sq = (size_t)n * n;
    void *h_A_f128, *h_x_f128, *h_y_hy_f128;
    allocate_and_init_f128(n, &h_A_f128, &h_x_f128, &h_y_hy_f128);

    half *d_sa_single, *d_sx_single; float *d_tmp_gpu;
    cudaMalloc(&d_sa_single, n_sq * 2);
    cudaMalloc(&d_sx_single, n * 2);
    cudaMalloc(&d_tmp_gpu, n * 4);

    double rho = 7.0; 
    std::vector<unsigned short> h_sa_all(n_sq * S_MAT); 
    std::vector<int> h_ta_all(n * S_MAT);
    std::vector<unsigned short> h_sx_all(n * S_VEC); 
    std::vector<int> h_tx_all(S_VEC);

    split_matrix_f128(n, S_MAT, h_A_f128, h_sa_all.data(), h_ta_all.data(), rho);
    split_vector_f128(n, S_VEC, h_x_f128, h_sx_all.data(), h_tx_all.data(), rho);

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);

    std::vector<float> h_tmp_gpu(n);
    for(int s = 0; s < S_MAT; s++) {
        cudaMemcpy(d_sa_single, h_sa_all.data() + (size_t)s*n_sq, n_sq*2, cudaMemcpyHostToDevice);
        for(int t = 0; t < S_VEC; t++) {
            cudaMemcpy(d_sx_single, h_sx_all.data() + (size_t)t*n, n*2, cudaMemcpyHostToDevice);
            float a=1.0f, b=0.0f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &a, d_sa_single, CUDA_R_16F, n, d_sx_single, CUDA_R_16F, n, &b, d_tmp_gpu, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            
            cudaMemcpy(h_tmp_gpu.data(), d_tmp_gpu, n * 4, cudaMemcpyDeviceToHost);
            for(int i=0; i<n; i++) {
                cpu_accumulate_f128(1, h_ta_all[s * n + i], h_tx_all[t], &h_tmp_gpu[i], (void*)((char*)h_y_hy_f128 + i * 16));
            }
        }
    }

    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float t_gpu; cudaEventElapsedTime(&t_gpu, st, ed);

    double res[2];
    cpu_final_eval(n, h_y_hy_f128, h_A_f128, h_x_f128, (double)t_gpu, res);
    printf("N=%-5d | RMSE: %.2e\n", n, res[1]);

    cudaFree(d_sa_single); cudaFree(d_sx_single); cudaFree(d_tmp_gpu);
    free_f128(h_A_f128, h_x_f128, h_y_hy_f128);
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    for(int n=128; n<=2048; n*=2) run_safe_test(n, h);
    cublasDestroy(h); return 0;
}