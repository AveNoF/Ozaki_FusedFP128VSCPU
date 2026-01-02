#ifndef HYBRID_RECON_HPP
#define HYBRID_RECON_HPP

#include <cuda_fp16.h>
#include <vector>

#ifndef __CUDACC__
    #include <quadmath.h>
    typedef __float128 float128_t;
#endif

extern "C" {
    void allocate_and_init_f128(int n, void** A_ptr, void** x_ptr, void** y_ptr);
    void free_f128(void* A_ptr, void* x_ptr, void* y_ptr);
    void split_matrix_f128(int n, int s_mat, const void* A_ptr, half* h_sa, int* h_ta, double rho);
    void split_vector_f128(int n, int s_vec, const void* v_ptr, half* h_sx, int* h_tx, double rho);
    void cpu_accumulate_f128(int n, const int* h_ta_row, int tx_val, const float* h_tmp_gpu, void* h_y_ptr);
    void cpu_final_eval(int n, const void* h_y_hy, const void* h_A_ptr, const void* h_x_ptr, double* results);
    // グラフ比較用：CPUでのナイーブ計算
    void cpu_naive_fp128(int n, const void* A, const void* x, void* y, double* time_ms);
}
#endif