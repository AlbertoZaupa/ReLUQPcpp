#include <cuda_runtime.h>
#include "linalg.hpp"
#include "solver.hpp"
#include <cmath>
#include <chrono>

void display_matrix(real* d_M, int rows, int cols);
void display_GPU_matrix(real* d_M, int rows, int cols);

// Custom kernel for computing matrix S in Solver::compute_matrices
__global__ void computeSKernel(const real* eigs, real rho, real* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1 / (1 + rho * eigs[i]);
    }
}

// Custom kernel to perform efficiently A = A + D * s, where D is diagonal
__global__ void addScaledDiagMatrixKernel(real* A, const real* d, real s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[n * i + i] += d[i] * s;
    }
}

QP_data::QP_data() : nx(0), nc(0), H(), A(), T(),
    M(), M_inv(), g(), l(), u(), eigs() {}

QP_data::QP_data(int nx_, int nc_, real* H_, real* A_, real* T_, real* M_, 
        real* M_inv_, real* g_, real* l_, real* u_, real* eigs_) : 
        nx(nx_), nc(nc_), H(nx_, nx_), A(nc_, nx_), T(nx_, nx_), M(nc_), 
        M_inv(nc_), g(nx_), l(nc_), u(nc_), eigs(nx_) {
    H.copyFromHost(H_);
    A.copyFromHost(A_);
    T.copyFromHost(T_);
    M.copyFromHost(M_);
    M_inv.copyFromHost(M_inv_);
    g.copyFromHost(g_);
    l.copyFromHost(l_);
    u.copyFromHost(u_);
    eigs.copyFromHost(eigs_);
}

ADMM_data::ADMM_data(real primal_res_, real dual_res_, real rho_) : primal_res(primal_res_),
    dual_res(dual_res_), rho(rho_) {}

Solver::Solver(bool verbose, int reactive_rho_duration_, real rho_, int nx, int nc, real* H_,
            real* A_, real* T_, real* M_, 
            real* M_inv_, real* g_, real* l_,
            real* u_, real* eigs_) :
            reactive_rho_duration(reactive_rho_duration_),
            qp_data(nx, nc, H_, A_, T_, M_, M_inv_, g_, l_, u_, eigs_),
            W(nx + 2*nc, nx + 2*nc), B(nx + nc, nx),
            b(nx + 2*nc), x(nx), z(nc), lambda(nc), z_prev(nc), 
            state1(nx + 2*nc), state2(nx + 2*nc), rho(rho_), MA(nc, nx),
            AtM(nx, nc), Tt(nx, nx), At(nx, nc), AT(nc, nx), 
            TtAt(nx, nc), TtAtM(nx, nc), TtAtMA(nx, nx),
            S(nx), rhoSTtAtMA(nx, nx), rhoSTtAtM(nx, nc), STtAt(nx, nc),
            STt(qp_data.nx, qp_data.nx), A__(nx, nx), B_(nx, nc), C_(nx, nc),
            D_(nc, nx), E_(nc, nc), F_(nc, nc), G_(nc, nx), H__(nc, nc),
            I_(nc, nc), B1_(nx, nx), B2_(nc, nx)
{
    // The ADMM state is intialized to zero
    state1.scale(0);
    state = &state1;
    next_state = &state2;
    // b is set to zero to avoid having to initialize its last block during ADMM execution
    b.scale(0);

    CUBLAS_CHECK(cublasCreate(&handle));

    left_diag_matmul(MA, qp_data.M, qp_data.A);
    MA.transpose(handle, AtM);
    qp_data.T.transpose(handle, Tt);
    qp_data.A.transpose(handle, At);
    matmul(handle, AT, qp_data.A, qp_data.T);
    matmul(handle, TtAt, Tt, At);
    right_diag_matmul(TtAtM, TtAt, qp_data.M);
    matmul(handle, TtAtMA, TtAtM, qp_data.A);
    eye(I_);

    // Display the total amount of memory allocated
    if (verbose) {
        real bytes = 0;
        bytes += B2_.rows * B2_.cols * sizeof(real);
        bytes += B1_.rows * B1_.cols * sizeof(real);
        bytes += I_.rows * I_.cols * sizeof(real);
        bytes += H__.rows * H__.cols * sizeof(real);
        bytes += F_.rows * F_.cols * sizeof(real);
        bytes += E_.rows * E_.cols * sizeof(real);
        bytes += D_.rows * D_.cols * sizeof(real);
        bytes += C_.rows * C_.cols * sizeof(real);
        bytes += B_.rows * B_.cols * sizeof(real);
        bytes += A__.rows * A__.cols * sizeof(real);
        bytes += STt.rows * STt.cols * sizeof(real);
        bytes += STtAt.rows * STtAt.cols * sizeof(real);
        bytes += rhoSTtAtM.rows * rhoSTtAtM.cols * sizeof(real);
        bytes += rhoSTtAtMA.rows * rhoSTtAtMA.cols * sizeof(real);
        bytes += S.n_elements * sizeof(real);
        bytes += qp_data.M.n_elements * sizeof(real);
        bytes += qp_data.M_inv.n_elements * sizeof(real);
        bytes += qp_data.eigs.n_elements * sizeof(real);
        bytes += W.rows * W.cols * sizeof(real);
        bytes += B.rows * B.cols * sizeof(real);
        bytes += b.n_elements * sizeof(real);
        bytes += MA.rows * MA.cols * sizeof(real);
        bytes += AtM.rows * AtM.cols * sizeof(real);
        bytes += Tt.rows * Tt.cols * sizeof(real);
        bytes += At.rows * At.cols * sizeof(real);
        bytes += AT.rows * AT.cols * sizeof(real);
        bytes += TtAt.rows * TtAt.cols * sizeof(real);
        bytes += TtAtM.rows * TtAtM.cols * sizeof(real);
        bytes += TtAtMA.rows * TtAtMA.cols * sizeof(real);
        std::cout << "CppSolver memory usage: " << bytes / 1e6 << " Mbs\n";
    }
}

void Solver::setup(real abs_tol_, int max_iter_, int check_interval_) {
    abs_tol = abs_tol_;
    max_iter_ = max_iter;
    check_interval = check_interval_;
    compute_matrices();
}

Solver::~Solver() {
    CUBLAS_CHECK(cublasDestroy(handle));
}

void Solver::compute_matrices() {
    // Computation of S = (I + rho * EIG) ^ {-1}
    int blockSize = 256;
    int gridSize = (state1.n_elements + blockSize - 1) / blockSize;
    computeSKernel<<<gridSize, blockSize>>>(qp_data.eigs.d_data, rho, S.d_data, state1.n_elements);
    CUDA_CHECK(cudaGetLastError());

    // Computation of auxiliary matrices
    left_diag_matmul(rhoSTtAtMA, S, TtAtMA);
    rhoSTtAtMA.scale(rho);
    left_diag_matmul(rhoSTtAtM, S, TtAtM);
    rhoSTtAtM.scale(rho);
    left_diag_matmul(STtAt, S, TtAt);
    left_diag_matmul(STt, S, Tt);

    // Computation of matrices A_, B_, C_, D_, E_, F_, G_, H_, I_
    // which constitute W as W = [[A_ B_ C_]; [D_ E_ F_]; [G_ H_ I_]]
    matmul_scale(handle, A__, qp_data.T, rhoSTtAtMA, -1);
    matmul_scale(handle, B_, qp_data.T, rhoSTtAtM, 2);
    matmul_scale(handle, C_, qp_data.T, STtAt, -1);
    matmul_scale_add(handle, D_, AT, rhoSTtAtMA, -1, qp_data.A);
    matmul_scale(handle, E_, AT, rhoSTtAtM, 2);
    E_.addScalarMatrix(-1);
    matmul_scale(handle, F_, AT, STtAt, -1);
    blockSize = 256,
    gridSize = (qp_data.nc + blockSize - 1) / blockSize;
    addScaledDiagMatrixKernel<<<gridSize, blockSize>>>(F_.d_data, qp_data.M_inv.d_data, 1 / rho, qp_data.nc);
    CUDA_CHECK(cudaGetLastError());
    G_.copyFromDevice(MA.d_data);
    G_.scale(rho);
    diag(H__, qp_data.M, -rho);

    // COMPUTATION OF W
    // Offsets in memory (column-major order)
    int col_offset_1 = 0;
    int col_offset_2 = qp_data.nx;
    int col_offset_3 = qp_data.nx + qp_data.nc;
        
    int row_offset_1 = 0;
    int row_offset_2 = qp_data.nx;
    int row_offset_3 = qp_data.nx + qp_data.nc;

    int ldW = qp_data.nx + 2 * qp_data.nc;
    int ldA = qp_data.nx;
    // Copy A_ to W(0,0)
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_1 + col_offset_1 * ldW,
                   ldW * sizeof(real),
                   A__.d_data,
                   ldA * sizeof(real),
                   ldA * sizeof(real),  // Width = num_rows in bytes
                    qp_data.nx, // Height = num_cols
                   cudaMemcpyDeviceToDevice));
    // Copy B_ to W(0, 1)
    int ldB = qp_data.nx;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_1 + col_offset_2 * ldW,
        ldW * sizeof(real),
        B_.d_data,
        ldB * sizeof(real),
        ldB * sizeof(real),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy C_ to W(0, 2)
    int ldC = qp_data.nx;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_1 + col_offset_3 * ldW,
        ldW * sizeof(real),
        C_.d_data,
        ldC * sizeof(real),
        ldC * sizeof(real),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy D_ to W(1, 0)
    int ldD = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_2 + col_offset_1 * ldW,
        ldW * sizeof(real),
        D_.d_data,
        ldD * sizeof(real),
        ldD * sizeof(real),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy E_ to W(1, 1)
    int ldE = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_2 + col_offset_2 * ldW,
        ldW * sizeof(real),
        E_.d_data,
        ldE * sizeof(real),
        ldE * sizeof(real),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy F_ to W(1, 2)
    int ldF = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_2 + col_offset_3 * ldW,
        ldW * sizeof(real),
        F_.d_data,
        ldF * sizeof(real),
        ldF * sizeof(real),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy D_ to W(2, 0)
    int ldG = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_3 + col_offset_1 * ldW,
        ldW * sizeof(real),
        G_.d_data,
        ldG * sizeof(real),
        ldG * sizeof(real),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy H_ to W(2, 1)
    int ldH = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_3 + col_offset_2 * ldW,
        ldW * sizeof(real),
        H__.d_data,
        ldH * sizeof(real),
        ldH * sizeof(real),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy I_ to W(2, 2)
    int ldI = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_3 + col_offset_3 * ldW,
        ldW * sizeof(real),
        I_.d_data,
        ldI * sizeof(real),
        ldI * sizeof(real),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
        
    // COMPUTATION OF B
    matmul_scale(handle, B1_, qp_data.T, STt, -1);
    matmul_scale(handle, B2_, AT, STt, -1);
    int ldB1 = qp_data.nx;
    int ldB2 = qp_data.nc;
    ldB = qp_data.nx + qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(B.d_data,
        ldB * sizeof(real),
        B1_.d_data,
        ldB1 * sizeof(real),
        ldB1 * sizeof(real),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(B.d_data + qp_data.nx,
        ldB * sizeof(real),
        B2_.d_data,
        ldB2 * sizeof(real),
        ldB2 * sizeof(real),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
        

    // COMPUTATION OF b
    matvecmul(handle, b.d_data, B, qp_data.g);
}

void Solver::forward_pass() {
    // state = W*state + b
    affine_transformation(handle, *next_state, W, *state, b);
    Vector *tmp = state;
    state = next_state;
    next_state = tmp;
    clip((*state).d_data + qp_data.nx, qp_data.l, qp_data.u);
}

void Solver::solve() {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    iter = 0;
    int i;
    real rho_new = rho;
    for (i = 0; i < max_iter; i++) {
        // state = Project( W*state + b )
        forward_pass();

        // If the residuals will be checked on the next iteration, copy the current
        // value of z into z_prev
        if ((i + 1) % check_interval == 0) {
            get_z(z_prev);
        }

        // Check residuals and update rho
        if (i > 0 && i % check_interval == 0) {
            ADMM_data data = compute_rho_residuals(rho_new);
            rho_new = data.rho;

            if (check_termination(data.primal_res, data.dual_res)) break;
            if (i / check_interval <= reactive_rho_duration) {
                rho = rho_new;
                compute_matrices();
            }
            else {
                if (rho_new > 5 * rho) {
                    rho = 5 * rho;
                    compute_matrices();
                } else if (rho_new < rho / 5) {
                    rho = rho / 5;
                    compute_matrices();
                }
            }
        }
    }
    iter = i;
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    solve_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000; // solve time in seconds
}

bool Solver::check_termination(real primal_res, real dual_res) {
    int nc = qp_data.nc;
    int nx = qp_data.nx;
    return primal_res < abs_tol * std::sqrt(nc) && dual_res < abs_tol * std::sqrt(nx);
}

ADMM_data Solver::compute_rho_residuals(real rho_) {
    get_x(x);
    get_z(z);
    get_lambda(lambda);
    Vector t1(qp_data.nc);
    matvecmul(handle, t1, qp_data.A, x);
    Vector t2(qp_data.nx);
    matvecmul(handle, t2, qp_data.H, x);
    Vector t3(qp_data.nx);
    matvecmul(handle, t3, At, lambda);
    Vector primal_res_vec(qp_data.nc);
    vecdiff(primal_res_vec, t1, z);
    Vector dual_res_vec(qp_data.nx);
    Vector tmp(qp_data.nc);
    vecdiff(tmp, z_prev, z);
    matvecmul(handle, dual_res_vec, AtM, tmp);
    Vector lagrangian_grad(qp_data.nx);
    vecsum(lagrangian_grad, t2, qp_data.g);
    vecsum(lagrangian_grad, t3, lagrangian_grad);
    
    double primal_res = infinity_norm(primal_res_vec);
    double dual_res = rho_ * infinity_norm(dual_res_vec);
    double lagrangian_grad_norm = infinity_norm(lagrangian_grad);
    double t1_norm = infinity_norm(t1);
    double z_norm = infinity_norm(z);
    double t2_norm = infinity_norm(t2);
    double g_norm = infinity_norm(qp_data.g);
    double t3_norm = infinity_norm(t3);

    double num = primal_res / std::max(t1_norm, z_norm);
    double den = lagrangian_grad_norm / std::max(std::max(t2_norm, t3_norm), g_norm);
    real rho_new = rho_ * std::sqrt(num / den);
    if (rho_new < 1e-6) rho_new = 1e-6;
    if (rho_new > 1e6) rho_new = 1e6;

    return ADMM_data(primal_res, dual_res, rho_new);
}
    
// Extract x, lambda, z from the state vector of ADMM. x, lambda, z are copied to 
// avoid having multiple references to the same memory location
void Solver::get_x(Vector &dst) {
    if (dst.n_elements != qp_data.nx) {
        std::cerr << "Function get_x. Vectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(dst.d_data, (*state).d_data, qp_data.nx * sizeof(real), cudaMemcpyDeviceToDevice));
}

void Solver::get_z(Vector &dst) {
    if (dst.n_elements != qp_data.nc) {
        std::cerr << "Function get_z. Vectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(dst.d_data, (*state).d_data + qp_data.nx, qp_data.nc * sizeof(real), cudaMemcpyDeviceToDevice));
}

void Solver::get_lambda(Vector &dst) {
    if (dst.n_elements != qp_data.nc) {
        std::cerr << "Function get_lambda. Vectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(dst.d_data, (*state).d_data + qp_data.nx + qp_data.nc, qp_data.nc * sizeof(real), cudaMemcpyDeviceToDevice));
}

std::vector<real> Solver::get_results() {
    get_x(x);
    std::vector<real> result(qp_data.nx);
    real* result_data = result.data();
    CUDA_CHECK(cudaMemcpy(result_data, x.d_data, qp_data.nx * sizeof(real), cudaMemcpyDeviceToHost));
    return result;
}

void Solver::update(const real* g, const real* l, const real* u, real rho_) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    if (g) qp_data.g.copyFromHost(g);
    if (l) qp_data.l.copyFromHost(l);
    if (u) qp_data.u.copyFromHost(u);

    // A negative value means that we want to warm start rho
    if (rho_ > 0) {
        rho = rho_;
        compute_matrices();
    } else {
        // b update
        matvecmul(handle, b.d_data, B, qp_data.g);
    }

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    update_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000; // solve time in seconds
}

void display_GPU_matrix(real* d_M, int rows, int cols) {
    real M[rows * cols];
    CUDA_CHECK(cudaMemcpy(M, d_M, rows * cols * sizeof(real), cudaMemcpyDeviceToHost));
    display_matrix(M, rows, cols);
}

void display_matrix(real* M, int rows, int cols) {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            std::cout << M[j * rows + i] << " ";
        }
        std::cout << "\n";
    }
} 

/*int main() {
    real H[9] = {6, 2, 1, 2, 5, 2, 1, 2, 4};
    real g[3] = {-8, -3, -3};
    real A[15] = {1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1};
    real l[5] = {3, 0, -10, -10, -10};
    real u[5] = {3, 0, INFINITY, INFINITY, INFINITY};
    real T[9] = {0.27741242, 0.27728448, -0.27724179, -0.30141514, 0.42981273, 0.03884762, 0.15799476, -0.12442978, 0.48464509};
    real eigs[3] = {2.30738285e-01, 2.88861322e+02, 5.43016374e+02};
    real M[5] = {1e3, 1e3, 1, 1, 1};
    real M_inv[5] = {1e-3, 1e-3, 1, 1, 1};
    int nx = 3;
    int nc = 5;
    real rho = 0.1;

    Solver solver(rho, nx, nc, H, A, T, M, M_inv, g, l, u, eigs);
    solver.setup();
    solver.solve();
    std::cout << "Solver execution time: " << solver.solve_time << " s" << std::endl;
    std::vector<real> result = solver.get_results();
    display_matrix(result.data(), nx, 1);

    return 0;
}*/