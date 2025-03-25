#include <cuda_runtime.h>
#include "linalg.hpp"
#include "solver.hpp"
#include <cmath>
#include <chrono>

void display_matrix(float* d_M, int rows, int cols);
void display_GPU_matrix(float* d_M, int rows, int cols);

// Custom kernel for computing matrix S in Solver::compute_matrices
__global__ void computeSKernel(const float* eigs, float rho, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1 / (1 + rho * eigs[i]);
    }
}

// Custom kernel to perform efficiently A = A + D * s, where D is diagonal
__global__ void addScaledDiagMatrixKernel(float* A, const float* d, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        A[n * i + i] += d[i] * s;
    }
}

QP_data::QP_data() : nx(0), nc(0), H(), A(), T(),
    M(), M_inv(), g(), l(), u(), eigs() {}

QP_data::QP_data(int nx_, int nc_, float* H_, float* A_, float* T_, float* M_, 
        float* M_inv_, float* g_, float* l_, float* u_, float* eigs_) : 
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

ADMM_data::ADMM_data(float primal_res_, float dual_res_, float rho_) : primal_res(primal_res_),
    dual_res(dual_res_), rho(rho_) {}

Solver::Solver(float rho_, int nx, int nc, float* H_,
            float* A_, float* T_, float* M_, 
            float* M_inv_, float* g_, float* l_,
            float* u_, float* eigs_) :
            qp_data(nx, nc, H_, A_, T_, M_, M_inv_, g_, l_, u_, eigs_),
            W(nx + 2*nc, nx + 2*nc), B(nx + nc, nx),
            b(nx + 2*nc), x(nx), z(nc), lambda(nc), z_prev(nc), 
            state(nx + 2*nc), rho(rho_), MA(nc, nx),
            AtM(nx, nc), Tt(nx, nx), At(nx, nc), AT(nc, nx), 
            TtAt(nx, nc), TtAtM(nx, nc), TtAtMA(nx, nx)
{
    // The ADMM state is intialized to zero
    state.scale(0);
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
}

void Solver::setup(float abs_tol_, int max_iter_, int check_interval_) {
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
    Vector S = Vector(qp_data.nx);
    int blockSize = 256;
    int gridSize = (state.n_elements + blockSize - 1) / blockSize;
    computeSKernel<<<gridSize, blockSize>>>(qp_data.eigs.d_data, rho, S.d_data, state.n_elements);
    CUDA_CHECK(cudaGetLastError());

    // Computation of auxiliary matrices
    Matrix rhoSTtAtMA(qp_data.nx, qp_data.nx);
    left_diag_matmul(rhoSTtAtMA, S, TtAtMA);
    rhoSTtAtMA.scale(rho);
    Matrix rhoSTtAtM(qp_data.nx, qp_data.nc);
    left_diag_matmul(rhoSTtAtM, S, TtAtM);
    rhoSTtAtM.scale(rho);
    Matrix STtAt(qp_data.nx, qp_data.nc);
    left_diag_matmul(STtAt, S, TtAt);
    Matrix STt(qp_data.nx, qp_data.nx);
    left_diag_matmul(STt, S, Tt);

    // Computation of matrices A_, B_, C_, D_, E_, F_, G_, H_, I_
    // which constitute W as W = [[A_ B_ C_]; [D_ E_ F_]; [G_ H_ I_]]

    Matrix A_(qp_data.nx, qp_data.nx);
    matmul_scale(handle, A_, qp_data.T, rhoSTtAtMA, -1);
    Matrix B_(qp_data.nx, qp_data.nc);
    matmul_scale(handle, B_, qp_data.T, rhoSTtAtM, 2);
    Matrix C_(qp_data.nx, qp_data.nc);
    matmul_scale(handle, C_, qp_data.T, STtAt, -1);
    Matrix D_(qp_data.nc, qp_data.nx);
    matmul_scale_add(handle, D_, AT, rhoSTtAtMA, -1, qp_data.A);
    Matrix E_(qp_data.nc, qp_data.nc);
    matmul_scale(handle, E_, AT, rhoSTtAtM, 2);
    E_.addScalarMatrix(-1);
        
    // Custom Kernel to perform F_ = F_ + D * s efficiently
    Matrix F_(qp_data.nc, qp_data.nc);
    matmul_scale(handle, F_, AT, STtAt, -1);
    blockSize = 256,
    gridSize = (qp_data.nc + blockSize - 1) / blockSize;
    addScaledDiagMatrixKernel<<<gridSize, blockSize>>>(F_.d_data, qp_data.M_inv.d_data, 1 / rho, qp_data.nc);
    CUDA_CHECK(cudaGetLastError());


    Matrix G_(qp_data.nc, qp_data.nx);
    G_.copyFromDevice(MA.d_data);
    G_.scale(rho);
    Matrix H_(qp_data.nc, qp_data.nc);
    diag(H_, qp_data.M, -rho);
    Matrix I_(qp_data.nc, qp_data.nc);
    eye(I_);

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
                   ldW * sizeof(float),
                   A_.d_data,
                   ldA * sizeof(float),
                   ldA * sizeof(float),  // Width = num_rows in bytes
                    qp_data.nx, // Height = num_cols
                   cudaMemcpyDeviceToDevice));
    // Copy B_ to W(0, 1)
    int ldB = qp_data.nx;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_1 + col_offset_2 * ldW,
        ldW * sizeof(float),
        B_.d_data,
        ldB * sizeof(float),
        ldB * sizeof(float),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy C_ to W(0, 2)
    int ldC = qp_data.nx;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_1 + col_offset_3 * ldW,
        ldW * sizeof(float),
        C_.d_data,
        ldC * sizeof(float),
        ldC * sizeof(float),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy D_ to W(1, 0)
    int ldD = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_2 + col_offset_1 * ldW,
        ldW * sizeof(float),
        D_.d_data,
        ldD * sizeof(float),
        ldD * sizeof(float),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy E_ to W(1, 1)
    int ldE = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_2 + col_offset_2 * ldW,
        ldW * sizeof(float),
        E_.d_data,
        ldE * sizeof(float),
        ldE * sizeof(float),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy F_ to W(1, 2)
    int ldF = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_2 + col_offset_3 * ldW,
        ldW * sizeof(float),
        F_.d_data,
        ldF * sizeof(float),
        ldF * sizeof(float),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy D_ to W(2, 0)
    int ldG = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_3 + col_offset_1 * ldW,
        ldW * sizeof(float),
        G_.d_data,
        ldG * sizeof(float),
        ldG * sizeof(float),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy H_ to W(2, 1)
    int ldH = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_3 + col_offset_2 * ldW,
        ldW * sizeof(float),
        H_.d_data,
        ldH * sizeof(float),
        ldH * sizeof(float),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    // Copy I_ to W(2, 2)
    int ldI = qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(W.d_data + row_offset_3 + col_offset_3 * ldW,
        ldW * sizeof(float),
        I_.d_data,
        ldI * sizeof(float),
        ldI * sizeof(float),  // Width = num_rows in bytes
        qp_data.nc, // Height = num_cols
        cudaMemcpyDeviceToDevice));
        
    // COMPUTATION OF B
    Matrix B1_(qp_data.nx, qp_data.nx);
    matmul_scale(handle, B1_, qp_data.T, STt, -1);
    Matrix B2_(qp_data.nc, qp_data.nx);
    matmul_scale(handle, B2_, AT, STt, -1);
    int ldB1 = qp_data.nx;
    int ldB2 = qp_data.nc;
    ldB = qp_data.nx + qp_data.nc;
    CUDA_CHECK(cudaMemcpy2D(B.d_data,
        ldB * sizeof(float),
        B1_.d_data,
        ldB1 * sizeof(float),
        ldB1 * sizeof(float),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy2D(B.d_data + qp_data.nx,
        ldB * sizeof(float),
        B2_.d_data,
        ldB2 * sizeof(float),
        ldB2 * sizeof(float),  // Width = num_rows in bytes
        qp_data.nx, // Height = num_cols
        cudaMemcpyDeviceToDevice));
        

    // COMPUTATION OF b
    matvecmul(handle, b.d_data, B, qp_data.g);
}

void Solver::forward_pass() {
    // state = W*state + b
    affine_transformation(handle, state, W, state, b);
    clip(state.d_data + qp_data.nx, qp_data.l, qp_data.u);
}

void Solver::solve() {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    iter = 0;
    int i;
    float rho_new = rho;
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

            if (rho_new > 5 * rho) {
                rho = 5 * rho;
                compute_matrices();
            } else if (rho_new < rho / 5) {
                rho = rho / 5;
                compute_matrices();
            }
        }
    }
    iter = i;
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    solve_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000; // solve time in seconds
}

bool Solver::check_termination(float primal_res, float dual_res) {
    int nc = qp_data.nc;
    int nx = qp_data.nx;
    return primal_res < abs_tol * std::sqrt(nc) && dual_res < abs_tol * std::sqrt(nx);
}

ADMM_data Solver::compute_rho_residuals(float rho_) {
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
    // !!!! DANGEROUS CODE
    dual_res_vec.n_elements = qp_data.nc;
    vecdiff(dual_res_vec, z_prev, z);
    matvecmul(handle, dual_res_vec.d_data, AtM, dual_res_vec);
    dual_res_vec.n_elements = qp_data.nx;
    // END OF DANGEROUS CODE
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
    float rho_new = rho_ * std::sqrt(num / den);
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
    CUDA_CHECK(cudaMemcpy(dst.d_data, state.d_data, qp_data.nx * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Solver::get_z(Vector &dst) {
    if (dst.n_elements != qp_data.nc) {
        std::cerr << "Function get_z. Vectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(dst.d_data, state.d_data + qp_data.nx, qp_data.nc * sizeof(float), cudaMemcpyDeviceToDevice));
}

void Solver::get_lambda(Vector &dst) {
    if (dst.n_elements != qp_data.nc) {
        std::cerr << "Function get_lambda. Vectors dimensions mismatch." << std::endl;
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaMemcpy(dst.d_data, state.d_data + qp_data.nx + qp_data.nc, qp_data.nc * sizeof(float), cudaMemcpyDeviceToDevice));
}

std::vector<float> Solver::get_results() {
    get_x(x);
    std::vector<float> result(qp_data.nx);
    float* result_data = result.data();
    CUDA_CHECK(cudaMemcpy(result_data, x.d_data, qp_data.nx * sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

void Solver::update(const float* g, const float* l, const float* u, float rho_) {
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    if (g) qp_data.g.copyFromHost(g);
    if (l) qp_data.l.copyFromHost(l);
    if (u) qp_data.u.copyFromHost(u);

    // A negative value means that we want to warm start rho
    if (rho_ > 0) {
        rho = rho_;
        compute_matrices();
    }

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    update_time = std::chrono::duration<double, std::milli>(end - start).count() / 1000; // solve time in seconds
}

void display_GPU_matrix(float* d_M, int rows, int cols) {
    float M[rows * cols];
    CUDA_CHECK(cudaMemcpy(M, d_M, rows * cols * sizeof(float), cudaMemcpyDeviceToHost));
    display_matrix(M, rows, cols);
}

void display_matrix(float* M, int rows, int cols) {
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            std::cout << M[j * rows + i] << " ";
        }
        std::cout << "\n";
    }
} 

/*int main() {
    float H[9] = {6, 2, 1, 2, 5, 2, 1, 2, 4};
    float g[3] = {-8, -3, -3};
    float A[15] = {1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1};
    float l[5] = {3, 0, -10, -10, -10};
    float u[5] = {3, 0, INFINITY, INFINITY, INFINITY};
    float T[9] = {0.27741242, 0.27728448, -0.27724179, -0.30141514, 0.42981273, 0.03884762, 0.15799476, -0.12442978, 0.48464509};
    float eigs[3] = {2.30738285e-01, 2.88861322e+02, 5.43016374e+02};
    float M[5] = {1e3, 1e3, 1, 1, 1};
    float M_inv[5] = {1e-3, 1e-3, 1, 1, 1};
    int nx = 3;
    int nc = 5;
    float rho = 0.1;

    Solver solver(rho, nx, nc, H, A, T, M, M_inv, g, l, u, eigs);
    solver.setup();
    solver.solve();
    std::cout << "Solver execution time: " << solver.solve_time << " s" << std::endl;
    std::vector<float> result = solver.get_results();
    display_matrix(result.data(), nx, 1);

    return 0;
}*/