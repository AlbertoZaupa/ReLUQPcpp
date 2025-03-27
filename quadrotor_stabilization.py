import RHC_controllers
import numpy as np
import matplotlib.pyplot as plt
import argparse


def is_controllable(A, B):
    # Reachability matrix
    n = A.shape[0]
    m = B.shape[1]
    R = np.zeros((n, n * m))
    AkB = B
    for i in range(n):
        R[:, (n - 1 - i) * m : (n - i) * m] = AkB
        AkB = A @ AkB
    return np.linalg.matrix_rank(R) == n

def is_stabilizable(A, B):
    n = A.shape[0]
    m = B.shape[1]
    eigenvalues, _ = np.linalg.eig(A)

    # PBH test
    for i, lam in enumerate(eigenvalues):
        if np.abs(lam) >= 1:
            PBH_matrix = np.hstack((lam * np.eye(n) - A, B))
            if np.linalg.matrix_rank(PBH_matrix) < n:
                return False
        
    return True

def rk4(f, x, u, dt):
    k1 = dt*f(x, u)
    k2 = dt*f(x + k1 / 2, u)
    k3 = dt*f(x + k2 / 2, u)
    k4 = dt*f(x + k3, u)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run controller with a solver flag.")
    parser.add_argument("--solver", type=str, required=True, help="Solver flag value (string).")
    parser.add_argument("--plot", type=int, required=True, help="Plot flag value (int).")
    args = parser.parse_args()
    SOLVER_FLAG = args.solver  # Use the passed argument
    PLOT_FLAG = args.plot != 0
    assert SOLVER_FLAG in ["reluqp", "osqp", "cppsolver", "reluqp_warm"]

    g = 9.81
    m = 1.5
    Jx = Jy = 0.03
    Jz = 0.06
    l = 0.225
    c = 0.015
    dt = 0.01
    N = 50

    def quadrotor_dynamics(state, inputs):
        T = np.sum(inputs)
        tau_x = l * (inputs[1] - inputs[3])
        tau_y = l * (inputs[2] - inputs[0])
        tau_z = c * (inputs[0] + inputs[2] - inputs[1] - inputs[3])

        phi = state[6]
        sphi = np.sin(phi)
        cphi = np.cos(phi)
        theta = state[8]
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        psi = state[10]
        spsi = np.sin(psi)
        cpsi = np.cos(psi)
        wx = state[7]
        wy = state[9]
        wz = state[11]
        vx = state[1]
        vy = state[3]
        vz = state[5]
        
        vx_dot = T * (spsi*sphi + cpsi*stheta*cphi) / m
        vy_dot = T * (spsi*stheta*cphi - cpsi*sphi) / m
        vz_dot = T * ctheta * cphi / m - g
        wx_dot = (Jy - Jz) / Jx * wy * wz + tau_x / Jx
        wy_dot = (Jz - Jx) / Jy * wz * wx + tau_y / Jy
        wz_dot = (Jx - Jy) / Jz * wx * wy + tau_z / Jz

        return np.array([vx, vx_dot, vy, vy_dot, vz, vz_dot, wx, wx_dot, wy, wy_dot, wz, wz_dot])
    

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
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1/m, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 1/Jx, 0, 0],
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

    Q = np.diag(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 100, 1, 100, 1, 100, 1]))
    R = np.eye(4) * 10
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
    M = 200
    K = np.zeros((B.shape[1], B.shape[0]))
    
    theta_x = np.zeros(M)
    theta_y = np.zeros(M)
    u = np.zeros((M, 4))

    # π / 6 tilt along all axes
    x0 = np.array([0, 0, 0, 0, 0, 0, np.pi/6, 0, -np.pi/6, 0, 0, 0])
    # Constant force applied to compensate gravity
    u_const = np.ones(4) * m * g / 4
    controller_class = None
    if SOLVER_FLAG in ["reluqp", "reluqp_warm"]:
        controller_class = RHC_controllers.ReLUQP_controller
    elif SOLVER_FLAG == "osqp":
        controller_class = RHC_controllers.OSQP_controller
    elif SOLVER_FLAG == "cppsolver":
        controller_class = RHC_controllers.CppSolver_controller
    
    solve_times = []
    controller = controller_class(x0, A, B, K, Q, R, Pf, N, Ex,
                                            dx, cx, Eu, du, cu, Ef, df, cf)
    x_t = x0.copy()

    for t in range(M):
        # Save states
        theta_x[t] = x_t[6]
        theta_y[t] = x_t[8]
        if t == 0:
            u_t = controller.solve(t)
        else:
            u_t = controller.solve(t, x_t)
        u_t += u_const
        u_t = np.clip(u_t, np.zeros(4), 4 * np.ones(4))
        # save input
        u[t, :] = u_t
        # propagate dynamics
        x_t = rk4(quadrotor_dynamics, x_t, u_t, dt)
    
    if SOLVER_FLAG == "reluqp_warm":
        print(f"{SOLVER_FLAG} average solve time: {(controller.solve_time - controller.worst_case_time) / (M - 1)}\n")
    else:
        print(f"{SOLVER_FLAG} average solve time: {controller.solve_time / M}\n")
    print(f"{SOLVER_FLAG} worst case solve time: {controller.worst_case_time}\n")
    #print(f"{SOLVER_FLAG} worst case solve time iteration: {controller.worst_case_time_iter}\n")


    if PLOT_FLAG:
        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.plot(np.array(range(M)) * dt, theta_x, color="royalblue", linestyle="-", linewidth=2, label=r"$\theta_x(t)$")
        ax.plot(np.array(range(M)) * dt, theta_y, color="tomato", linestyle="--", linewidth=2, label=r"$\theta_y(t)$")
        ax.set_ylabel(r"$\theta(t)$")
        ax.set_xlabel('t')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"/shared/quadrotor_angles.png")

        plt.style.use("ggplot")
        fig, ax = plt.subplots()
        ax.plot(np.array(range(M)) * dt, u[:, 0], color="royalblue", linestyle="-", linewidth=2, label=r"$u_1(t)$")
        ax.plot(np.array(range(M)) * dt, u[:, 1], color="tomato", linestyle="--", linewidth=2, label=r"$u_2(t)$")
        ax.plot(np.array(range(M)) * dt, u[:, 2], color="seagreen", linestyle="-", linewidth=2, label=r"$u_3(t)$")
        ax.plot(np.array(range(M)) * dt, u[:, 3], color="purple", linestyle="--", linewidth=2, label=r"$u_4(t)$")
        ax.set_ylabel(r"$u(t)$")
        ax.set_xlabel('t')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"/shared/quadrotor_inputs.png")

