#ifndef HYBRID_RECON_HPP
#define HYBRID_RECON_HPP

#include <cuda_fp16.h>
#include <vector>

// CPU側(g++)でのみ __float128 を定義
#ifndef __CUDACC__
    #include <quadmath.h>
    typedef __float128 float128_t;
#endif

extern "C" {
    // 33桁精度のベクトル生成と分割
    void split_vector_f128(int n, int s_vec, void* v_ptr, half* h_sx, int* h_tx, double rho);

    // 高精度再構成と誤差評価
    void cpu_reconstruct_and_eval(
        int n, int s_mat, int s_vec,
        const float* h_tmpc, const int* h_ta, const int* h_tx,
        const double* h_A, const void* h_x_ptr,
        double t_gpu_ms, double* results
    );
}
#endif