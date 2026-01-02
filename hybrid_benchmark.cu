#include "hybrid_reconstruction.hpp"
#include <cublas_v2.h>
#include <iostream>
#include <vector>

#define S_MAT 16 
#define S_VEC 16
#define MAX_THREADS 1024

__global__ void dev_split_A(double* A_res, half* d_sa_single, int* d_ta_single, int n, double rho) {
    extern __shared__ double sdata[];
    int row = blockIdx.x; int tid = threadIdx.x;
    if (row >= n) return;
    size_t row_offset = (size_t)row * n;
    
    double local_max = 0.0;
    for (int j = tid; j < n; j += blockDim.x) local_max = fmax(local_max, fabs(A_res[row_offset + j]));
    sdata[tid] = local_max; __syncthreads();
    for (int st = blockDim.x/2; st > 0; st /= 2) { if(tid < st) sdata[tid] = fmax(sdata[tid], sdata[tid+st]); __syncthreads(); }

    __shared__ int tau;
    if (tid == 0) {
        int t; frexp(sdata[0], &t);
        d_ta_single[row] = t; tau = t;
    }
    __syncthreads();

    double sigma = scalbn(0.75, (int)rho + tau);
    for (int j = tid; j < n; j += blockDim.x) {
        size_t idx = row_offset + j;
        double val = A_res[idx];
        double q = (val + sigma) - sigma;
        half tmp = __double2half(scalbn(q, -tau));
        d_sa_single[idx] = tmp;
        A_res[idx] = val - scalbn((double)tmp, tau);
    }
}

void run_safe_test(int n, cublasHandle_t handle) {
    size_t n_sq = (size_t)n * n;
    std::vector<double> h_A(n_sq);
    for(size_t i=0; i<n_sq; i++) h_A[i] = (double)rand()/RAND_MAX;
    void* h_x = malloc(n * 16);

    double *d_A_res; half *d_sa_single, *d_sx_all; int *d_ta_single, *d_tx_all; float *d_tmpc;
    cudaMalloc(&d_A_res, n_sq * 8); cudaMalloc(&d_sa_single, n_sq * 2);
    cudaMalloc(&d_sx_all, (size_t)S_VEC * n * 2); cudaMalloc(&d_ta_single, n * 4);
    cudaMalloc(&d_tx_all, S_VEC * 4); cudaMalloc(&d_tmpc, (size_t)S_MAT * S_VEC * n * 4);

    cudaMemcpy(d_A_res, h_A.data(), n_sq * 8, cudaMemcpyHostToDevice);
    double rho = 7.0; 

    std::vector<half> h_sa_all(n_sq * S_MAT);
    std::vector<int> h_ta_all(n * S_MAT);
    for(int s=0; s<S_MAT; s++) {
        dev_split_A<<<n, MAX_THREADS, MAX_THREADS*8>>>(d_A_res, d_sa_single, d_ta_single, n, rho);
        cudaMemcpy(h_sa_all.data() + (size_t)s*n_sq, d_sa_single, n_sq*2, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ta_all.data() + (size_t)s*n, d_ta_single, n*4, cudaMemcpyDeviceToHost);
    }

    std::vector<half> h_sx(n * S_VEC); std::vector<int> h_tx(S_VEC);
    split_vector_f128(n, S_VEC, h_x, h_sx.data(), h_tx.data(), rho);
    cudaMemcpy(d_sx_all, h_sx.data(), (size_t)n * S_VEC * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tx_all, h_tx.data(), S_VEC * 4, cudaMemcpyHostToDevice);

    cudaEvent_t st, ed; cudaEventCreate(&st); cudaEventCreate(&ed); cudaEventRecord(st);
    for(int s=0; s<S_MAT; s++) {
        cudaMemcpy(d_sa_single, h_sa_all.data() + (size_t)s*n_sq, n_sq*2, cudaMemcpyHostToDevice);
        for(int t=0; t<S_VEC; t++) {
            float a=1.0f, b=0.0f;
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, 1, n, &a, d_sa_single, CUDA_R_16F, n, d_sx_all + (size_t)t*n, CUDA_R_16F, n, &b, d_tmpc + (size_t)(s*S_VEC+t)*n, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
    }
    cudaEventRecord(ed); cudaEventSynchronize(ed);
    float t_gpu; cudaEventElapsedTime(&t_gpu, st, ed);

    std::vector<float> h_tmpc((size_t)S_MAT*S_VEC*n);
    cudaMemcpy(h_tmpc.data(), d_tmpc, h_tmpc.size()*4, cudaMemcpyDeviceToHost);
    double res[2];
    cpu_reconstruct_and_eval(n, S_MAT, S_VEC, h_tmpc.data(), h_ta_all.data(), h_tx.data(), h_A.data(), h_x, t_gpu, res);
    
    printf("N=%-5d | Time: %8.2f ms | RMSE: %.2e\n", n, res[0], res[1]);

    cudaFree(d_A_res); cudaFree(d_sa_single); cudaFree(d_sx_all); cudaFree(d_ta_single); cudaFree(d_tx_all); cudaFree(d_tmpc); free(h_x);
}

int main() {
    cublasHandle_t h; cublasCreate(&h);
    printf("--- 🛡️ Reliable Ozaki Engine (Safe Zone: N <= 4096) ---\n");
    for(int n=128; n<=4096; n*=2) run_safe_test(n, h);
    cublasDestroy(h); cudaDeviceReset();
    return 0;
}