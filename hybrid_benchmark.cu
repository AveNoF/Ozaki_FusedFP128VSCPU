#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "hybrid_reconstruction.hpp"

using namespace std;

#define S_MAT 14
#define S_VEC 14
#define BLOCK_SIZE 1024

__global__ void dev_fused_split_A(double* A_res, half* split_A_all, int* tau_A_map, int s, int n, double rho) {
    __shared__ double sdata[BLOCK_SIZE];
    int row = blockIdx.x; int tid = threadIdx.x;
    if (row >= n) return;
    size_t row_offset = (size_t)row * n;
    double local_max = 0.0;
    for (int j = tid; j < n; j += blockDim.x) local_max = fmax(local_max, fabs(A_res[row_offset + j]));
    sdata[tid] = local_max; __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) sdata[tid] = fmax(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    __shared__ int tau;
    if (tid == 0) { frexp(sdata[0], &tau); tau_A_map[s * n + row] = tau; } __syncthreads();
    double sigma = 0.75 * exp2(rho + tau);
    for (int j = tid; j < n; j += blockDim.x) {
        size_t idx = row_offset + j;
        double val = A_res[idx];
        double q = (val + sigma) - sigma;
        half tmp = __double2half(scalbn(q, -tau));
        split_A_all[(size_t)s * n * n + idx] = tmp;
        A_res[idx] = val - scalbn((double)tmp, tau);
    }
}

__global__ void dev_split_x(double* x_res, half* split_x_all, int* tau_x_map, int t, int n, double rho) {
    if (threadIdx.x == 0) {
        double mx = 0.0;
        for (int k = 0; k < n; k++) mx = fmax(mx, fabs(x_res[k]));
        int tx; frexp(mx, &tx);
        tau_x_map[t] = tx;
    }
    __syncthreads();
    int tx = tau_x_map[t];
    double sigma = 0.75 * exp2(rho + tx);
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        double val = x_res[j];
        double q = (val + sigma) - sigma;
        half tmp = __double2half(scalbn(q, -tx));
        split_x_all[(size_t)t * n + j] = tmp;
        x_res[j] = val - scalbn((double)tmp, tx);
    }
}

void run_test(int n, ofstream &csv, cublasHandle_t handle) {
    size_t n_sq = (size_t)n * n;
    vector<double> h_A(n_sq), h_x(n);
    for (size_t i = 0; i < n_sq; i++) h_A[i] = (double)rand()/RAND_MAX;
    for (size_t i = 0; i < n; i++) h_x[i] = (double)rand()/RAND_MAX;

    double *d_A, *d_x, *d_y; half *d_sa, *d_sx; int *d_ta, *d_tx; float *d_tmpc;
    cudaMalloc(&d_A, n_sq * sizeof(double));
    cudaMalloc(&d_x, n * sizeof(double));
    cudaMalloc(&d_y, n * sizeof(double));
    cudaMalloc(&d_sa, (size_t)S_MAT * n_sq * sizeof(half));
    cudaMalloc(&d_sx, (size_t)S_VEC * n * sizeof(half));
    cudaMalloc(&d_ta, (size_t)S_MAT * n * sizeof(int));
    cudaMalloc(&d_tx, (size_t)S_VEC * sizeof(int));
    cudaMalloc(&d_tmpc, (size_t)S_MAT * S_VEC * n * sizeof(float));

    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    double al = 1.0, be = 0.0;

    // Warm-up
    cublasDgemv(handle, CUBLAS_OP_T, n, n, &al, d_A, n, d_x, 1, &be, d_y, 1);

    // cuBLAS Measure
    cudaMemcpy(d_A, h_A.data(), n_sq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(s);
    cublasDgemv(handle, CUBLAS_OP_T, n, n, &al, d_A, n, d_x, 1, &be, d_y, 1);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float t_cub; cudaEventElapsedTime(&t_cub, s, e);

    // Hybrid GPU Part
    double rho = ceil((53.0 - (24.0 - log2((double)n))/2.0));
    cudaEventRecord(s);
    for(int i=0; i<S_MAT; i++) dev_fused_split_A<<<n, 1024>>>(d_A, d_sa, d_ta, i, n, rho);
    for(int i=0; i<S_VEC; i++) dev_split_x<<<1, 1024>>>(d_x, d_sx, d_tx, i, n, rho);
    for(int i=0; i<S_MAT; i++){
        float alf=1.0f, bef=0.0f;
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, S_VEC, n, &alf, d_sa+(size_t)i*n_sq, CUDA_R_16F, n, d_sx, CUDA_R_16F, n, &bef, d_tmpc+(size_t)i*S_VEC*n, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(e); cudaEventSynchronize(e);
    float t_gpu; cudaEventElapsedTime(&t_gpu, s, e);

    vector<float> h_tmpc((size_t)S_MAT * S_VEC * n);
    vector<int> h_ta_v(S_MAT * n), h_tx_v(S_VEC);
    cudaMemcpy(h_tmpc.data(), d_tmpc, h_tmpc.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ta_v.data(), d_ta, h_ta_v.size()*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_tx_v.data(), d_tx, h_tx_v.size()*4, cudaMemcpyDeviceToHost);

    double hy_ms, cp_ms, hy_rmse, cp_rmse, cub_rmse;
    cpu_hybrid_reconstruct_and_eval(n, S_MAT, S_VEC, h_tmpc.data(), h_ta_v.data(), h_tx_v.data(), h_A.data(), h_x.data(), (double)t_gpu, (double)t_cub, hy_ms, cp_ms, hy_rmse, cp_rmse, cub_rmse);

    csv << n << "," << t_cub << "," << cp_ms << "," << hy_ms << "," << cub_rmse << "," << cp_rmse << "," << hy_rmse << "," << S_MAT << endl;

    cudaFree(d_A); cudaFree(d_x); cudaFree(d_y); cudaFree(d_sa); cudaFree(d_sx); cudaFree(d_ta); cudaFree(d_tx); cudaFree(d_tmpc);
}

int main() {
    cublasHandle_t handle; cublasCreate(&handle);
    ofstream csv("results.csv");
    csv << "N,cuBLAS_T,CPU128_T,Hybrid_T,cuBLAS_R,CPU128_R,Hybrid_R,SplitNum" << endl;
    for (int n = 32; n <= 16384; n *= 2) {
        cout << "Running N = " << n << "..." << endl;
        run_test(n, csv, handle);
    }
    csv.close();
    cublasDestroy(handle);
    return 0;
}