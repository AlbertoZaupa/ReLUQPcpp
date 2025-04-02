#ifndef RELUQP_HPP
#define RELUQP_HPP

#include "linalg.hpp"
#include <iostream>
#include <cstdlib>
#include <vector>

class QP_data {
public:
    int nx, nc;
    Matrix H, A, T;
    Vector M, M_inv, g, l, u, eigs;
    QP_data();
    QP_data(int nx_, int nc_, float* H_, float* A_, float* T_, float* M_, 
        float* M_inv_, float* g_, float* l_, float* u_, float* eigs_);
};

class ADMM_data {
public:
    float primal_res, dual_res, rho;
    ADMM_data(float primal_res_, float dual_res_, float rho_);
};

class Solver {
public:
    Solver(int reactive_rho_duration, float rho_, int nx, int nc, float* H_,
        float* A_, float* T_, float* M_, 
        float* M_inv_, float* g_, float* l_,
        float* u_, float* eigs_);
    ~Solver();
    void solve();
    void setup(float abs_tol_, int max_iter_, int check_interval_);
    void update(const float* g, const float* l, const float* u, float rho_);
    std::vector<float> get_results();
    float solve_time = 0;
    float update_time = 0;
    int iter = 0;
private:
    QP_data qp_data;
    Vector state1, state2, x, z, lambda, z_prev;
    Vector* state;
    Vector* next_state;
    Matrix W, B;
    Vector b;
    Matrix TtAtMA, TtAtM, TtAt, Tt, At, AT, MA, AtM;
    cublasHandle_t handle;
    float rho;
    float abs_tol = 1e-3;
    int max_iter = 4000;
    int check_interval = 25;
    int reactive_rho_duration = 0;

    void compute_matrices();
    void forward_pass();
    bool check_termination(float primal_res, float dual_res);
    ADMM_data compute_rho_residuals(float rho_);
    void get_x(Vector &dst);
    void get_z(Vector &dst);
    void get_lambda(Vector &dst);
};

#endif
