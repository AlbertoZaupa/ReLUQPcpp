import RHC_controllers
import numpy as np
import scipy
    

if __name__ == '__main__':
    g = 9.81
    m = 1.5
    Jx = Jy = 0.03
    Jz = 0.06
    l = 0.225
    c = 0.035
    dt = 0.01
    N = 20

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

    Q = np.diag(np.array([1, 0, 1, 0, 1, 0, 1, 0.05, 1, 0.05, 1, 0.05]))
    R = np.eye(4)
    Pf = Q
    Ex = np.eye(12)
    Ef = Ex
    dx = np.ones((12,)) * np.inf
    cx = - np.ones((12,)) * np.inf
    df = dx
    cf = cx
    Eu = np.eye(4)
    du = np.array([4, 4, 4, 4])
    cu = np.array([0, 0, 0, 0])

    M = 20
    
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    # π / 6 tilt along all axes
    x0 = np.array([0, 0, 0, 0, 0, 0, np.pi/6, 0, np.pi/6, 0, np.pi/6, 0])
    reluqp_controller = RHC_controllers.ReLUQP_controller(x0, A, B, Q, R, Pf, N, Ex,
                                                      dx, cx, Eu, du, cu, Ef, df, cf)
    # ReLUQP simulation
    for t in range(M):
        if t == 0:
            u_t = reluqp_controller.solve()
        else:
            u_t = reluqp_controller.solve(x0)
        x0 = A @ x0 + B @ (-K @ x0 + u_t)
    print(f"ReLUQP total solve time: {reluqp_controller.solve_time}")

    x0 = np.array([0, 0, 0, 0, 0, 0, np.pi/6, 0, np.pi/6, 0, np.pi/6, 0])
    osqp_controller = RHC_controllers.OSQP_controller(x0, A, B, Q, R, Pf, N, Ex,
                                                      dx, cx, Eu, du, cu, Ef, df, cf)
    # OSQP simulation
    for t in range(M):
        if t == 0:
            u_t = osqp_controller.solve()
        else:
            u_t = osqp_controller.solve(x0)
        x0 = A @ x0 + B @ (-K @ x0 + u_t)
    print(f"OSQP total solve time: {osqp_controller.solve_time}")

    x0 = np.array([0, 0, 0, 0, 0, 0, np.pi/6, 0, np.pi/6, 0, np.pi/6, 0])
    pysolver_controller = RHC_controllers.PySolver_controller(x0, A, B, Q, R, Pf, N, Ex,
                                                      dx, cx, Eu, du, cu, Ef, df, cf)
    # PySolver simulation
    for t in range(M):
        if t == 0:
            u_t = pysolver_controller.solve()
        else:
            u_t = pysolver_controller.solve(x0)
        x0 = A @ x0 + B @ (-K @ x0 + u_t)
    print(f"PySolver total solve time: {pysolver_controller.solve_time}")


    # fig, ax = plt.subplots()
    # ax.plot(x[:, 0], x[:, 1], marker='o', markersize=3, linestyle='-', color='blue')
    # ax.set_xlabel(r"$x_1\left(t\right)$")
    # ax.set_ylabel(r"$x_2\left(t\right)$")
    # ax.set_title('State trajectory')
    # ax.grid(True)
    # ax.axis('equal')  # Ensure the scale is equal on both axes
    # plt.tight_layout()
    # plt.savefig(f"results/double_integrator_benchmark/state_trajectory.png")

    # fig, ax = plt.subplots()
    # ax.plot(range(M),u, marker='o', markersize=3, linestyle='-', color='blue')
    # ax.set_xlabel(r"$t$")
    # ax.set_ylabel(r"$u\left(t\right)$")
    # ax.set_title('Input signal')
    # ax.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"results/double_integrator_benchmark/control_signal.png")