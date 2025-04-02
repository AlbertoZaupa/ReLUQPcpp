import numpy as np
import argparse
from solver_wrapper import CppSolver
from reluqp import ReLU_QP
from osqp import OSQP
from utils import qp_initialization
from scipy import sparse
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run controller with a solver flag.")
    parser.add_argument("--solver", type=str, required=True, help="Solver flag value (string).")
    args = parser.parse_args()
    SOLVER_FLAG = args.solver  # Use the passed argument
    assert SOLVER_FLAG in ["reluqp", "osqp", "cppsolver", "reluqp_warm"]

    M = 300
    # Loading the data
    data = np.load('atlas_balancing_output.npz')
    H = data['H']
    g = data['g']
    A = data['A']
    l = data['l']
    u = data['u']
    x_ref = data['x_ref']
    X_ref = np.repeat(x_ref[:, np.newaxis], M, axis=1)
    X = data['X']
    dx = X - X_ref
    g_x0 = data['g_x0']
    lu_x0 = data['lu_x0']

    H = np.ascontiguousarray(H)
    A = np.ascontiguousarray(A)
    g = np.ascontiguousarray(g)
    l = np.ascontiguousarray(l)
    u = np.ascontiguousarray(u)
    g_x0 = np.ascontiguousarray(g_x0)
    lu_x0 = np.ascontiguousarray(lu_x0)
    dx = np.ascontiguousarray(dx)

    if SOLVER_FLAG in ["reluqp", "reluqp_warm"]:
        solver = ReLU_QP()
        solver.setup(H, g, A, l, u, precision=torch.float64)
    elif SOLVER_FLAG == "cppsolver":
        T, eigs, M_, M_inv_ = qp_initialization(H, A, l, u)
        solver = CppSolver(0.1, H.shape[0], A.shape[0], H, A, T, M_, M_inv_, g, l, u, eigs)
        solver.setup()
    elif SOLVER_FLAG == "osqp":
        solver = OSQP()
        solver.setup(P=sparse.csc_matrix(H), q=g, A=sparse.csc_matrix(A), l=l, u=u, eps_abs=1e-3, eps_rel=0, verbose=False)

    solve_time = 0
    worst_case_time = -1
    for t in range(M):
        dx_t = dx[:, t]
        if SOLVER_FLAG in ["reluqp", "reluqp_warm"]:
            solver.update(g=g + g_x0 @ dx_t, l=l + lu_x0 @ dx_t, u=u + lu_x0 @ dx_t)
        elif SOLVER_FLAG == "cppsolver":
            solver.update(g=g + g_x0 @ dx_t, l=l + lu_x0 @ dx_t, u=u + lu_x0 @ dx_t, rho=-1)
        elif SOLVER_FLAG == "osqp":
            solver.update(q=g + g_x0 @ dx_t, l=l + lu_x0 @ dx_t, u=u + lu_x0 @ dx_t)
        result = solver.solve()
        solve_time += result.info.solve_time
        if result.info.solve_time > worst_case_time:
            worst_case_time = result.info.solve_time
        #print(f"Solve time: {result.info.solve_time}")
        #print(f"Iterations: {result.info.iter}")

    if SOLVER_FLAG == "reluqp_warm":
        print(f"{SOLVER_FLAG} average solve time: {(solve_time - worst_case_time) / (M - 1)}")
    else:
        print(f"{SOLVER_FLAG} average solve time: {solve_time / M}")
    print(f"{SOLVER_FLAG} worst case time: {worst_case_time}")
