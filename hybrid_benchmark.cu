#include "hybrid_reconstruction.hpp"
#include <cublas_v2.h>
#include <iostream>
#include <vector>

#define S_MAT 16
#define S_VEC 16

void run_safe_test(int n, cublasHandle_t handle) {
    size_t n_sq = (size_t)n * n;
    void *h_A_f128, *h_x_f128;
    allocate_and_init_f128(n, &h_A_f128, &h_x_f128);

    half *d_sa_single, *d_sx_all; float *d_tmpc;
    cudaMalloc(&d_sa_single, n_sq * 2);
    cudaMalloc(&d_sx_all, (size_t)S_VEC * n * 2);
    cudaMalloc(&d_tmpc, (size_t)S_MAT * S_VEC * n * 4);

    double rho = 7.0; 
    std::vector<half> h_sa_all(n_sq * S_MAT); 
    std::vector<int> h_ta_all(n * S_MAT);      // 行列用 ta: S * n
    std::vector<half> h_sx(n * S_VEC); 
    std::vector<int> h_tx(S_VEC);              // ベクトル用 tx: S 個

    printf("N=%d | High-Prec Splitting...\n", n);
    split_matrix_f128(n, S_MAT, h_A_f128, h_sa_all.data(), h_ta_all.data(), rho);
    split_vector_f128(n, S_VEC, h_x_f128, h_sx.data(), h_tx.data(), rho);

    cudaMemcpy(d_sx_all, h_sx.data(), (size_t)n * S_VEC * 2, cudaMemcpyHostToDevice);

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed);
    cudaEventRecord(st);
    for(int s = 0; s < S_MAT; s++) {
        cudaMemcpy(d_sa_single, h_sa_all.data() + (size_t)s*n_sq, n_sq*2, cudaMemcpyHostToDevice);
        for(int t = 0; t < S_VEC; t++) {
            float a=1.0f, b=0.0f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &a, d_sa_single, CUDA_R_16F, n, d_sx_all + (size_t)t*n, CUDA_R_16F, n, &b, d_tmpc + (size_t)(s*S_VEC+t)*n, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float t_gpu; cudaEventElapsedTime(&t_gpu, st, ed);

    std::vector<float> h_tmpc((size_t)S_MAT * S_VEC * n);
    cudaMemcpy(h_tmpc.data(), d_tmpc, h_tmpc.size() * 4, cudaMemcpyDeviceToHost);
    
    double res[2];
    cpu_reconstruct_and_eval(n, S_MAT, S_VEC, h_tmpc.data(), h_ta_all.data(), h_tx.data(), h_A_f128, h_x_f128, (double)t_gpu, res);
    printf("N=%-5d | Time: %8.2f ms | RMSE: %.2e\n", n, res[0], res[1]);

    cudaFree(d_sa_single); cudaFree(d_sx_all); cudaFree(d_tmpc);
    free_f128(h_A_f128, h_x_f128);
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    printf("--- Ozaki Engine Safe Sanctuary (RTX 2060) ---\n");
    for(int n=128; n<=4096; n*=2) run_safe_test(n, h);
    cublasDestroy(h); cudaDeviceReset(); return 0;
}