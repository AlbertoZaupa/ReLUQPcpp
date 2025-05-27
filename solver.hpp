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
    QP_data(int nx_, int nc_, real* H_, real* A_, real* T_, real* M_, 
        real* M_inv_, real* g_, real* l_, real* u_, real* eigs_);
};

class ADMM_data {
public:
    real primal_res, dual_res, rho;
    ADMM_data(real primal_res_, real dual_res_, real rho_);
};

class Solver {
public:
    Solver(bool verbose, int reactive_rho_duration, real rho_, int nx, int nc, real* H_,
        real* A_, real* T_, real* M_, 
        real* M_inv_, real* g_, real* l_,
        real* u_, real* eigs_);
    ~Solver();
    void solve();
    void setup(real abs_tol_, int max_iter_, int check_interval_);
    void update(const real* g, const real* l, const real* u, real rho_);
    std::vector<real> get_results();
    real solve_time = 0;
    real update_time = 0;
    int iter = 0;
private:
    QP_data qp_data;
    Vector state1, state2, x, z, lambda, z_prev;
    Vector* state;
    Vector* next_state;
    Matrix W, B;
    Vector b;
    Vector S;
    Matrix rhoSTtAtMA, rhoSTtAtM, STtAt, STt, A__, B_, C_, D_, E_, F_, G_, H__, I_, B1_, B2_;
    Matrix TtAtMA, TtAtM, TtAt, Tt, At, AT, MA, AtM;
    cublasHandle_t handle;
    real rho;
    real abs_tol = 1e-3;
    int max_iter = 4000;
    int check_interval = 25;
    int reactive_rho_duration = 0;

    void compute_matrices();
    void forward_pass();
    bool check_termination(real primal_res, real dual_res);
    ADMM_data compute_rho_residuals(real rho_);
    void get_x(Vector &dst);
    void get_z(Vector &dst);
    void get_lambda(Vector &dst);
};

#endif
