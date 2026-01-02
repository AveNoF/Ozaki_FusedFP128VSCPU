#include "hybrid_reconstruction.hpp"
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>

using namespace std;

void allocate_and_init_f128(int n, void** A_ptr, void** x_ptr, void** y_ptr) {
    size_t n_sq = (size_t)n * n;
    float128_t* A = (float128_t*)malloc(n_sq * 16);
    float128_t* x = (float128_t*)malloc(n * 16);
    float128_t* y = (float128_t*)malloc(n * 16);
    for(size_t i=0; i<n_sq; i++) A[i] = (float128_t)rand()/RAND_MAX + (float128_t)rand()/RAND_MAX * 1e-18Q;
    for(int i=0; i<n; i++) { x[i] = (float128_t)rand()/RAND_MAX + (float128_t)rand()/RAND_MAX * 1e-18Q; y[i] = 0.0Q; }
    *A_ptr = (void*)A; *x_ptr = (void*)x; *y_ptr = (void*)y;
}

void free_f128(void* A_ptr, void* x_ptr, void* y_ptr) { free(A_ptr); free(x_ptr); free(y_ptr); }

void split_matrix_f128(int n, int s_mat, const void* A_ptr, half* h_sa, int* h_ta, double rho) {
    const float128_t* A = (const float128_t*)A_ptr;
    vector<float128_t> res((size_t)n * n);
    memcpy(res.data(), A, (size_t)n * n * 16);
    for (int s = 0; s < s_mat; s++) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            float128_t mx = 0;
            for(int j=0; j<n; j++) mx = fmaxq(mx, fabsq(res[(size_t)i*n+j]));
            int ta = 0; if(mx > 0) frexpq(mx, &ta);
            h_ta[s * n + i] = ta;
            float128_t sigma = scalbnq(1.0Q, ta + (int)rho);
            for (int j = 0; j < n; j++) {
                size_t idx = (size_t)i * n + j;
                float128_t q = (res[idx] + sigma) - sigma;
                h_sa[(size_t)s * (size_t)n * n + idx] = __double2half((double)scalbnq(q, -ta));
                res[idx] -= q;
            }
        }
    }
}

void split_vector_f128(int n, int s_vec, const void* v_ptr, half* h_sx, int* h_tx, double rho) {
    const float128_t* v = (const float128_t*)v_ptr;
    vector<float128_t> res(n);
    memcpy(res.data(), v, n * 16);
    for (int t = 0; t < s_vec; t++) {
        float128_t mx = 0;
        for(int i=0; i<n; i++) mx = fmaxq(mx, fabsq(res[i]));
        int tx = 0; if(mx > 0) frexpq(mx, &tx);
        h_tx[t] = tx;
        float128_t sigma = scalbnq(1.0Q, tx + (int)rho);
        for (int i = 0; i < n; i++) {
            float128_t q = (res[i] + sigma) - sigma;
            h_sx[(size_t)t * n + i] = __double2half((double)scalbnq(q, -tx));
            res[i] -= q;
        }
    }
}

void cpu_accumulate_f128(int n, const int* h_ta_row, int tx_val, const float* h_tmp_gpu, void* h_y_ptr) {
    float128_t* y = (float128_t*)h_y_ptr;
    for(int i=0; i<n; i++) y[i] += scalbnq((float128_t)h_tmp_gpu[i], h_ta_row[i] + tx_val);
}

void cpu_calc_truth_fp256(int n, const void* A_ptr, const void* x_ptr, void* y_truth_ptr) {
    const float128_t* A = (const float128_t*)A_ptr;
    const float128_t* x = (const float128_t*)x_ptr;
    float128_t* y_out = (float128_t*)y_truth_ptr;
    #pragma omp parallel
    {
        mpfr_t ma, mx, msum, mprod;
        mpfr_inits2(256, ma, mx, msum, mprod, (mpfr_ptr)0);
        char buf[128];
        #pragma omp for
        for(int i=0; i<n; i++) {
            mpfr_set_zero(msum, 1);
            for(int j=0; j<n; j++) {
                quadmath_snprintf(buf, sizeof(buf), "%.40Qe", A[(size_t)i*n+j]); mpfr_set_str(ma, buf, 10, MPFR_RNDN);
                quadmath_snprintf(buf, sizeof(buf), "%.40Qe", x[j]); mpfr_set_str(mx, buf, 10, MPFR_RNDN);
                mpfr_mul(mprod, ma, mx, MPFR_RNDN); mpfr_add(msum, msum, mprod, MPFR_RNDN);
            }
            quadmath_snprintf(buf, sizeof(buf), "%.45Re", msum); y_out[i] = strtoflt128(buf, NULL);
        }
        mpfr_clears(ma, mx, msum, mprod, (mpfr_ptr)0);
    }
}

double measure_rmse_mpfr(int n, const void* A_ptr, const void* x_ptr, const void* y_test_ptr) {
    const float128_t* A = (const float128_t*)A_ptr;
    const float128_t* x = (const float128_t*)x_ptr;
    const float128_t* y_test = (const float128_t*)y_test_ptr;
    mpfr_t sum_err_sq, ma, mx, mprod, mtruth, mtest, mdiff;
    mpfr_inits2(256, sum_err_sq, ma, mx, mprod, mtruth, mtest, mdiff, (mpfr_ptr)0);
    mpfr_set_zero(sum_err_sq, 1);
    char buf[128];
    for(int i=0; i<n; i++) {
        mpfr_set_zero(mtruth, 1);
        for(int j=0; j<n; j++) {
            quadmath_snprintf(buf, sizeof(buf), "%.40Qe", A[(size_t)i*n+j]); mpfr_set_str(ma, buf, 10, MPFR_RNDN);
            quadmath_snprintf(buf, sizeof(buf), "%.40Qe", x[j]); mpfr_set_str(mx, buf, 10, MPFR_RNDN);
            mpfr_mul(mprod, ma, mx, MPFR_RNDN); mpfr_add(mtruth, mtruth, mprod, MPFR_RNDN); // 修正: msum -> mtruth
        }
        quadmath_snprintf(buf, sizeof(buf), "%.40Qe", y_test[i]); mpfr_set_str(mtest, buf, 10, MPFR_RNDN);
        mpfr_sub(mdiff, mtruth, mtest, MPFR_RNDN); mpfr_mul(mdiff, mdiff, mdiff, MPFR_RNDN); mpfr_add(sum_err_sq, sum_err_sq, mdiff, MPFR_RNDN);
    }
    mpfr_div_ui(sum_err_sq, sum_err_sq, n, MPFR_RNDN); mpfr_sqrt(sum_err_sq, sum_err_sq, MPFR_RNDN);
    double rmse = mpfr_get_d(sum_err_sq, MPFR_RNDN);
    mpfr_clears(sum_err_sq, ma, mx, mprod, mtruth, mtest, mdiff, (mpfr_ptr)0);
    return rmse;
}

void cpu_benchmark_baselines(int n, const void* A_ptr, const void* x_ptr, const void* y_truth_ptr, 
                             double* t128, double* err128, double* t64, double* err64) {
    const float128_t* A = (const float128_t*)A_ptr;
    const float128_t* x = (const float128_t*)x_ptr;
    double s128 = omp_get_wtime();
    vector<float128_t> y128(n, 0.0Q);
    for(int i=0; i<n; i++) for(int j=0; j<n; j++) y128[i] += A[(size_t)i*n+j] * x[j];
    *t128 = (omp_get_wtime() - s128) * 1000.0;
    *err128 = measure_rmse_mpfr(n, A_ptr, x_ptr, y128.data());
    double s64 = omp_get_wtime();
    vector<double> y64(n, 0.0);
    for(int i=0; i<n; i++) for(int j=0; j<n; j++) y64[i] += (double)A[(size_t)i*n+j] * (double)x[j];
    *t64 = (omp_get_wtime() - s64) * 1000.0;
    vector<float128_t> y64_f128(n); for(int i=0; i<n; i++) y64_f128[i] = (float128_t)y64[i];
    *err64 = measure_rmse_mpfr(n, A_ptr, x_ptr, y64_f128.data());
}