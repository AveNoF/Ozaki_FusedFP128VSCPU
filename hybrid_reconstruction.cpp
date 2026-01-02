#include "hybrid_reconstruction.hpp"
#include <cmath>
#include <omp.h>
#include <iostream>
#include <algorithm>

using namespace std;

void split_vector_f128(int n, int s_vec, void* v_ptr, half* h_sx, int* h_tx, double rho) {
    float128_t* v = (float128_t*)v_ptr;
    vector<float128_t> v_res(n);
    
    // 33桁精度のテストデータを生成
    for(int i=0; i<n; i++) {
        v[i] = (float128_t)rand()/RAND_MAX + 
               ((float128_t)rand()/RAND_MAX) * 1e-16Q + 
               ((float128_t)rand()/RAND_MAX) * 1e-32Q;
        v_res[i] = v[i];
    }

    for (int t = 0; t < s_vec; t++) {
        float128_t mx = 0;
        for(int i=0; i<n; i++) if(fabsq(v_res[i]) > mx) mx = fabsq(v_res[i]);
        int tx = 0;
        if(mx > 0) frexpq(mx, &tx);
        h_tx[t] = tx;

        float128_t sigma = scalbnq(0.75Q, (int)rho + tx);
        for (int i = 0; i < n; i++) {
            float128_t q = (v_res[i] + sigma) - sigma;
            h_sx[(size_t)t * n + i] = __double2half((double)scalbnq(q, -tx));
            v_res[i] -= scalbnq((float128_t)h_sx[(size_t)t * n + i], tx);
        }
    }
}

void cpu_reconstruct_and_eval(int n, int s_mat, int s_vec, const float* h_tmpc, const int* h_ta, const int* h_tx, const double* h_A, const void* h_x_ptr, double t_gpu_ms, double* out) {
    const float128_t* h_x = (const float128_t*)h_x_ptr;
    vector<float128_t> y_hy(n, 0.0Q);
    double t_start = omp_get_wtime();
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float128_t row_sum = 0.0Q;
        for (int s = 0; s < s_mat; s++) {
            int ta = h_ta[s * n + i];
            for (int t = 0; t < s_vec; t++) {
                size_t idx = (size_t)(s * s_vec + t) * n + i;
                row_sum += scalbnq((float128_t)h_tmpc[idx], ta + h_tx[t]);
            }
        }
        y_hy[i] = row_sum;
    }
    double t_recon = (omp_get_wtime() - t_start) * 1000.0;

    vector<float128_t> y_tr(n, 0.0Q);
    #pragma omp parallel for
    for(int i=0; i<n; i++) {
        for(int j=0; j<n; j++) y_tr[i] += (float128_t)h_A[(size_t)i*n+j] * h_x[j];
    }

    float128_t err_sq = 0;
    for(int i=0; i<n; i++) {
        float128_t d = y_tr[i] - y_hy[i];
        err_sq += d * d;
    }
    out[0] = t_gpu_ms + t_recon;
    out[1] = (double)sqrtq(err_sq / (float128_t)n);
}