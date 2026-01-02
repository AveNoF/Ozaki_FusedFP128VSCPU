#ifndef HYBRID_RECON_HPP
#define HYBRID_RECON_HPP

#include <cuda_fp16.h>
#include <vector>

#ifndef __CUDACC__
    #include <quadmath.h>
    typedef __float128 float128_t;
#endif

extern "C" {
    // 33桁データの生成とメモリ確保 (CPU)
    void allocate_and_init_f128(int n, void** A_ptr, void** x_ptr);
    void free_f128(void* A_ptr, void* x_ptr);

    // 33桁行列とベクトルの Ozaki 分割 (CPU)
    // 行列は行ごとに指数を、ベクトルは全体で1つの指数を持つように修正
    void split_matrix_f128(int n, int s_mat, const void* A_ptr, half* h_sa, int* h_ta, double rho);
    void split_vector_f128(int n, int s_vec, const void* v_ptr, half* h_sx, int* h_tx, double rho);

    // 33桁再構成と評価 (CPU)
    void cpu_reconstruct_and_eval(
        int n, int s_mat, int s_vec,
        const float* h_tmpc, const int* h_ta, const int* h_tx,
        const void* h_A_ptr, const void* h_x_ptr,
        double t_gpu_ms, double* results
    );
}

#endif