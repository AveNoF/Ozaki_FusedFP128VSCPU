#include "hybrid_reconstruction.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <vector>

using namespace std;

// 汎用的な Ozaki 分割関数 (128bit 対応)
void split_common_f128(int n, int size, int s_count, const float128_t* src, half* dst_h, int* dst_t, double rho) {
    vector<float128_t> res(size);
    for(int i=0; i<size; i++) res[i] = src[i];

    for (int s = 0; s < s_count; s++) {
        #pragma omp parallel for
        for (int i = 0; i < (size == n ? n : n); i++) {
            float128_t mx = 0;
            if (size == n) { // ベクトルの場合
                for(int k=0; k<n; k++) if(fabsq(res[k]) > mx) mx = fabsq(res[k]);
            } else { // 行列の場合 (行ごとの最大値)
                for(int j=0; j<n; j++) if(fabsq(res[(size_t)i*n+j]) > mx) mx = fabsq(res[(size_t)i*n+j]);
            }
            
            int tx = 0;
            if(mx > 0) frexpq(mx, &tx);
            dst_t[s * n + i] = tx;

            float128_t sigma = scalbnq(0.75Q, (int)rho + tx);
            for (int j = 0; j < (size == n ? 1 : n); j++) {
                size_t idx = (size == n) ? i : (size_t)i * n + j;
                float128_t q = (res[idx] + sigma) - sigma;
                dst_h[(size_t)s * size + idx] = __double2half((double)scalbnq(q, -tx));
                res[idx] -= scalbnq((float128_t)dst_h[(size_t)s * size + idx], tx);
            }
        }
    }
}

void split_matrix_f128(int n, int s_mat, const void* A_ptr, half* h_sa, int* h_ta, double rho) {
    split_common_f128(n, (size_t)n*n, s_mat, (const float128_t*)A_ptr, h_sa, h_ta, rho);
}

void split_vector_f128(int n, int s_vec, const void* v_ptr, half* h_sx, int* h_tx, double rho) {
    split_common_f128(n, n, s_vec, (const float128_t*)v_ptr, h_sx, h_tx, rho);
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
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) y_tr[i] += h_A[(size_t)i*n+j] * h_x[j];
    }

    float128_t err_sq = 0;
    for(int i=0; i<n; i++) {
        float128_t d = y_tr[i] - y_hy[i];
        err_sq += d * d;
    }
    out[0] = t_gpu_ms + t_recon;
    out[1] = (double)sqrtq(err_sq / (float128_t)n);
}