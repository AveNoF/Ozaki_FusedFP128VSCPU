#ifndef HYBRID_RECON_HPP
#define HYBRID_RECON_HPP

#include <cuda_fp16.h>
#include <vector>

#ifndef __CUDACC__
    #include <quadmath.h>
    #include <mpfr.h>
    typedef __float128 float128_t;
#endif

extern "C" {
    void allocate_and_init_f128(int n, void** A_ptr, void** x_ptr, void** y_ptr);
    void free_f128(void* A_ptr, void* x_ptr, void* y_ptr);
    void split_matrix_f128(int n, int s_mat, const void* A_ptr, half* h_sa, int* h_ta, double rho);
    void split_vector_f128(int n, int s_vec, const void* v_ptr, half* h_sx, int* h_tx, double rho);
    void cpu_accumulate_f128(int n, const int* h_ta_row, int tx_val, const float* h_tmp_gpu, void* h_y_ptr);
    double measure_rmse_mpfr(int n, const void* A, const void* x, const void* y_test);
    void cpu_calc_truth_fp256(int n, const void* A, const void* x, void* y_truth);
    void cpu_benchmark_baselines(int n, const void* A, const void* x, const void* y_truth, 
                                 double* t128, double* err128, double* t64, double* err64);
}
#endif