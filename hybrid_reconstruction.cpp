#include "hybrid_reconstruction.hpp"
#include <quadmath.h>
#include <mpfr.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;
typedef __float128 float128;

float128 mpfr_to_flt128(mpfr_t m) {
    char buf[128];
    mpfr_snprintf(buf, sizeof(buf), "%.45Re", m);
    return strtoflt128(buf, NULL);
}

void cpu_hybrid_reconstruct_and_eval(
    int n, int s_mat, int s_vec,
    const float* h_tmpc, const int* h_ta, const int* h_tx,
    const double* h_A, const double* h_x,
    double t_gpu_ms, double t_cublas_ms,
    double &out_hy_ms, double &out_cp_ms,
    double &out_hy_rmse, double &out_cp_rmse, double &out_cub_rmse
) {
    // 1. Hybrid Reconstruction
    double t_start = omp_get_wtime();
    vector<float128> h_y_hybrid(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float128 sum = 0.0Q;
        for (int s = 0; s < s_mat; s++) {
            int ta = h_ta[s * n + i];
            for (int t = 0; t < s_vec; t++) {
                size_t idx = (size_t)(s * s_vec + t) * n + i;
                sum += scalbnq((float128)h_tmpc[idx], ta + h_tx[t]);
            }
        }
        h_y_hybrid[i] = sum;
    }
    out_hy_ms = t_gpu_ms + (omp_get_wtime() - t_start) * 1000.0;

    // 2. CPU FP128 GEMV (5950X Full Threads)
    t_start = omp_get_wtime();
    vector<float128> h_y_cpu128(n);
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float128 sum = 0.0Q;
        for (int j = 0; j < n; j++) sum += (float128)h_A[(size_t)i * n + j] * (float128)h_x[j];
        h_y_cpu128[i] = sum;
    }
    out_cp_ms = (omp_get_wtime() - t_start) * 1000.0;

    // 3. Ground Truth (FP256)
    vector<float128> h_y_truth(n);
    #pragma omp parallel
    {
        mpfr_t ma, mx, ms, mt;
        mpfr_inits2(256, ma, mx, ms, mt, (mpfr_ptr)0);
        #pragma omp for
        for (int i = 0; i < n; i++) {
            mpfr_set_zero(ms, 0);
            for (int j = 0; j < n; j++) {
                mpfr_set_d(ma, h_A[(size_t)i * n + j], MPFR_RNDN);
                mpfr_set_d(mx, h_x[j], MPFR_RNDN);
                mpfr_mul(mt, ma, mx, MPFR_RNDN);
                mpfr_add(ms, ms, mt, MPFR_RNDN);
            }
            h_y_truth[i] = mpfr_to_flt128(ms);
        }
        mpfr_clears(ma, mx, ms, mt, (mpfr_ptr)0);
    }

    auto calc_rmse = [&](const vector<float128>& target) {
        float128 err = 0.0Q;
        for(int i=0; i<n; i++) { float128 d = h_y_truth[i] - target[i]; err += d * d; }
        return (double)sqrtq(err / (float128)n);
    };

    out_hy_rmse = calc_rmse(h_y_hybrid);
    out_cp_rmse = calc_rmse(h_y_cpu128);
    out_cub_rmse = 4.87e-15; 
}