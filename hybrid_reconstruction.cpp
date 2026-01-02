#include "hybrid_reconstruction.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

// 128bit データの生成
void allocate_and_init_f128(int n, void** A_ptr, void** x_ptr) {
    size_t n_sq = (size_t)n * n;
    float128_t* A = (float128_t*)malloc(n_sq * 16);
    float128_t* x = (float128_t*)malloc(n * 16);
    for(size_t i=0; i<n_sq; i++) A[i] = (float128_t)rand()/RAND_MAX + (float128_t)rand()/RAND_MAX * 1e-18Q;
    for(int i=0; i<n; i++) x[i] = (float128_t)rand()/RAND_MAX + (float128_t)rand()/RAND_MAX * 1e-18Q;
    *A_ptr = (void*)A;
    *x_ptr = (void*)x;
}

void free_f128(void* A_ptr, void* x_ptr) {
    free(A_ptr); free(x_ptr);
}

// 分割ロジック (128bit)
void split_common_f128(int n, size_t size, int s_count, const float128_t* src, half* dst_h, int* dst_t, double rho) {
    vector<float128_t> res(size);
    for(size_t i=0; i<size; i++) res[i] = src[i];

    for (int s = 0; s < s_count; s++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            float128_t mx = 0;
            if (size == (size_t)n) { // ベクトル
                for(int k=0; k<n; k++) if(fabsq(res[k]) > mx) mx = fabsq(res[k]);
            } else { // 行列 (行ごとの最大値)
                for(int j=0; j<n; j++) if(fabsq(res[(size_t)i*n+j]) > mx) mx = fabsq(res[(size_t)i*n+j]);
            }
            int tx = 0;
            if(mx > 0) frexpq(mx, &tx);
            dst_t[s * n + i] = tx;

            float128_t sigma = scalbnq(0.75Q, (int)rho + tx);
            for (size_t j = 0; j < (size == (size_t)n ? 1 : n); j++) {
                size_t idx = (size == (size_t)n) ? i : (size_t)i * n + j;
                float128_t q = (res[idx] + sigma) - sigma;
                dst_h[s * size + idx] = __double2half((double)scalbnq(q, -tx));
                res[idx] -= scalbnq((float128_t)dst_h[s * size + idx], tx);
            }
        }
    }
}

void split_matrix_f128(int n, int s_mat, const void* A_ptr, half* h_sa, int* h_ta, double rho) {
    split_common_f128(n, (size_t)n*n, s_mat, (const float128_t*)A_ptr, h_sa, h_ta, rho);
}

void split_vector_f128(int n, int s_vec, const void* v_ptr, half* h_sx, int* h_tx, double rho) {
    split_common_f128(n, (size_t)n, s_vec, (const float128_t*)v_ptr, h_sx, h_tx, rho);
}

void cpu_reconstruct_and_eval(int n, int s_mat, int s_vec, const float* h_tmpc, const int* h_ta, const int* h_tx, const void* h_A_ptr, const void* h_x_ptr, double t_gpu_ms, double* out) {
    const float128_t* h_A = (const float128_t*)h_A_ptr;
    const float128_t* h_x = (const float128_t*)h_x_ptr;
    vector<float128_t> y_hy(n, 0.0Q);
    double t_start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int s = 0; s < s_mat; s++) {
            int ta = h_ta[s * n + i];
            for (int t = 0; t < s_vec; t++) {
                y_hy[i] += scalbnq((float128_t)h_tmpc[(size_t)(s * s_vec + t) * n + i], ta + h_tx[t]);
            }
        }
    }
    double t_recon = (omp_get_wtime() - t_start) * 1000.0;
    vector<float128_t> y_tr(n, 0.0Q);
    #pragma omp parallel for
    for(int i=0; i<n; i++) for(int j=0; j<n; j++) y_tr[i] += h_A[(size_t)i*n+j] * h_x[j];

    float128_t err_sq = 0;
    for(int i=0; i<n; i++) { float128_t d = y_tr[i] - y_hy[i]; err_sq += d * d; }
    out[0] = t_gpu_ms + t_recon; out[1] = (double)sqrtq(err_sq / (float128_t)n);
}