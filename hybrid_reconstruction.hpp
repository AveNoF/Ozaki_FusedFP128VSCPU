#ifndef HYBRID_RECON_HPP
#define HYBRID_RECON_HPP

#include <vector>

void cpu_hybrid_reconstruct_and_eval(
    int n, int s_mat, int s_vec,
    const float* h_tmpc, const int* h_ta, const int* h_tx,
    const double* h_A, const double* h_x,
    double t_gpu_ms, double t_cublas_ms,
    double &out_hy_ms, double &out_cp_ms,
    double &out_hy_rmse, double &out_cp_rmse, double &out_cub_rmse
);

#endif