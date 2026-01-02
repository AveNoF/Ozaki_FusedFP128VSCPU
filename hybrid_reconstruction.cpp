#include "hybrid_reconstruction.hpp"
#include <cmath>
#include <omp.h>
#include <cstdlib>
#include <cstring>

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
    size_t n_sq = (size_t)n * n;
    vector<float128_t> res(n_sq);
    memcpy(res.data(), A_ptr, n_sq * 16);
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
                h_sa[(size_t)s * n_sq + idx] = __double2half((double)scalbnq(q, -ta));
                res[idx] -= q; // [cite: 155]
            }
        }
    }
}

void split_vector_f128(int n, int s_vec, const void* v_ptr, half* h_sx, int* h_tx, double rho) {
    vector<float128_t> res(n);
    memcpy(res.data(), v_ptr, n * 16);
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

void cpu_final_eval(int n, const void* h_y_hy, const void* h_A_ptr, const void* h_x_ptr, double* out) {
    const float128_t* y_hy = (const float128_t*)h_y_hy;
    const float128_t* A = (const float128_t*)h_A_ptr;
    const float128_t* x = (const float128_t*)h_x_ptr;
    vector<float128_t> y_tr(n, 0.0Q);
    #pragma omp parallel for
    for(int i=0; i<n; i++) for(int j=0; j<n; j++) y_tr[i] += A[(size_t)i*n+j] * x[j];
    float128_t err_sq = 0;
    for(int i=0; i<n; i++) { float128_t d = y_tr[i] - y_hy[i]; err_sq += d * d; }
    out[0] = (double)sqrtq(err_sq / (float128_t)n);
}

void cpu_naive_fp128(int n, const void* A_ptr, const void* x_ptr, void* y_ptr, double* time_ms) {
    const float128_t* A = (const float128_t*)A_ptr;
    const float128_t* x = (const float128_t*)x_ptr;
    float128_t* y = (float128_t*)y_ptr;
    double start = omp_get_wtime();
    #pragma omp parallel for
    for(int i=0; i<n; i++) {
        float128_t sum = 0;
        for(int j=0; j<n; j++) sum += A[(size_t)i*n+j] * x[j];
        y[i] = sum;
    }
    *time_ms = (omp_get_wtime() - start) * 1000.0;
}