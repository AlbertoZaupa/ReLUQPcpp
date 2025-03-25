import RHC_controllers
import numpy as np
import scipy
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run controller with a solver flag.")
    parser.add_argument("--solver", type=str, required=True, help="Solver flag value (string).")
    args = parser.parse_args()
    SOLVER_FLAG = args.solver  # Use the passed argument
    assert SOLVER_FLAG in ["reluqp", "osqp", "cppsolver"]

    g = 9.81
    m = 1.5
    Jx = Jy = 0.03
    Jz = 0.06
    l = 0.225
    c = 0.035
    dt = 0.01
    N = 50

    A_ = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    B__ = np.array([[0, 0, 0, 0],
                    [0, 1/Jx, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1/Jy, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1/m],
                    [0, 0, 0, 0],
                    [1/Jx, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 1/Jy, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 1/Jz]])
    M = np.array([[1, 1, 1, 1],
                  [0, l, 0, -l],
                  [-l, 0, l, 0],
                  [c, -c, c, -c]])
    B_ = B__ @ M
    A = np.eye(12) + dt * A_
    B = dt * B_

    Q = np.diag(np.array([1, 0, 1, 0, 1, 0, 1, 0.01, 1, 0.01, 1, 0.01]))
    R = np.eye(4)
    Pf = Q
    Ex = np.eye(12)
    Ef = Ex
    dx = np.ones((12,)) * np.inf
    cx = - np.ones((12,)) * np.inf
    df = dx
    cf = cx
    Eu = np.eye(4)
    du = np.ones(4) * (4 - g*m/4)
    cu = np.ones(4) * (-g*m/4)
    M = 50
    
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    # π / 6 tilt along all axes
    x0 = np.array([0, 0, 0, 0, 0, 0, np.pi/6, 0, np.pi/6, 0, np.pi/6, 0])
    controller_class = None
    if SOLVER_FLAG == "reluqp":
        controller_class = RHC_controllers.ReLUQP_controller
    elif SOLVER_FLAG == "osqp":
        controller_class = RHC_controllers.OSQP_controller
    elif SOLVER_FLAG == "cppsolver":
        controller_class = RHC_controllers.CppSolver_controller
    
    solve_times = []
    for _ in range(1):
        x_t = x0.copy()
        controller = controller_class(x0, A, B, K, Q, R, Pf, N, Ex,
                                                      dx, cx, Eu, du, cu, Ef, df, cf)
        for t in range(M):
            if t == 0:
                u_t = controller.solve()
            else:
                u_t = controller.solve(x_t)
            x_t = A @ x_t + B @ (-K @ x_t + u_t)

        solve_times.append(controller.solve_time)
    
    print(f"{SOLVER_FLAG} mean solve time: {np.mean(solve_times)}")