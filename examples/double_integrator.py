import numpy as np
import scipy
import argparse
from .RHC_controllers import *
import matplotlib.pyplot as plt
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run controller with a solver flag.")
    parser.add_argument("--solver", type=str, required=True, help="Solver flag value (string).")
    parser.add_argument("--plot", type=int, required=True, help="Plot flag value (int).")
    parser.add_argument("--n", type=int, required=True, help="Horizon length (int).")
    parser.add_argument("--verbose", type=int, required=True, help="Verbose flag (int)")
    args = parser.parse_args()
    SOLVER_FLAG = args.solver  # Use the passed argument
    PLOT_FLAG = args.plot != 0
    N = args.n
    verbose = args.verbose != 0
    assert SOLVER_FLAG in ["reluqp", "osqp", "gpusolver",
                            "reluqp_warm", "gpusolver_1", "cpusolver"]

    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    B_ = B.reshape(-1)
    Q = np.eye(2)
    R = 1
    Pf = Q
    M = 20
    Ex = np.eye(2)
    Ef = Ex
    Eu = np.eye(1)
    dx = np.array([35, 30])
    cx = np.array([-35, -30])
    df = dx
    cf = cx
    du = np.array([1])
    cu = np.array([-1])

    x0 = np.array([10, 5])
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    x = np.zeros((M, 2))
    u = np.zeros((M,))

    controller_class = None
    if SOLVER_FLAG in ["reluqp", "reluqp_warm"]:
        controller_class = ReLUQP_controller
    elif SOLVER_FLAG == "osqp":
        controller_class = OSQP_controller
    elif SOLVER_FLAG == "gpusolver":
        controller_class = GpuSolver_controller
    
    solve_times = []
    controller = controller_class(x0, A, B, K, Q, R, Pf, N, Ex,
                                            dx, cx, Eu, du, cu, Ef, df, cf, verbose)
    for t in range(M):
        # Save states
        x[t, :] = x0
        if t == 0:
            u_t = controller.solve(t)
        else:
            u_t = controller.solve(t, x0)
        # save input
        u[t] = (-K @ x0 + u_t)[0]
        # propagate dynamics
        x0 = A @ x0 + B_ * u[t]
    
    if SOLVER_FLAG == "reluqp_warm":
        print(f"average solve time: {(controller.solve_time) / (M - 2)}")
    else:
        print(f"average solve time: {controller.solve_time / M}")
    print(f"worst case time: {controller.worst_case_time}")

    if PLOT_FLAG:
        fig, ax = plt.subplots()
        ax.plot(x[:, 0], x[:, 1], color="tomato", linestyle="-", linewidth=2)
        ax.set_ylabel(r"$x(t)$")
        ax.set_xlabel('t')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"/shared/di_states.pdf")

        fig, ax = plt.subplots()
        ax.plot(np.array(range(M)), u, color="royalblue", linestyle="-", linewidth=2, label=r"$u(t)$")
        ax.set_ylabel(r"$u(t)$")
        ax.set_xlabel('t')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"/shared/di_control.pdf")
  