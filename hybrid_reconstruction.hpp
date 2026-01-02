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

    // CPU側で即座に float128 精度で集計する関数
    void cpu_accumulate_f128(int n, int ta, int tx, const float* h_tmp_gpu, void* h_y_ptr);
    
    // 最終評価
    void cpu_final_eval(int n, const void* h_y_hy, const void* h_A_ptr, const void* h_x_ptr, double t_total, double* results);
}
#endif