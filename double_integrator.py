import RHC_controllers
import numpy as np
import scipy
    

if __name__ == '__main__':
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    B_ = B.reshape(-1)
    Q = np.eye(2)
    R = 1
    Pf = Q
    N = 50
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

    x0 = np.array([10, 5, 10, 5, 10, 5])
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
    x = np.zeros((M, 6))
    u = np.zeros((M, 3))

    reluqp_controller = RHC_controllers.ReLUQP_controller(x0[:2], A, B, Q, R, Pf, N, Ex,
                                                      dx, cx, Eu, du, cu, Ef, df, cf)
    osqp_controller = RHC_controllers.OSQP_controller(x0[2:4], A, B, Q, R, Pf, N, Ex,
                                                      dx, cx, Eu, du, cu, Ef, df, cf)
    pysolver_controller = RHC_controllers.PySolver_controller(x0[4:], A, B, Q, R, Pf, N, Ex,
                                                      dx, cx, Eu, du, cu, Ef, df, cf)
    # ReLUQP simulation
    for t in range(M):
        x[t, 0:2] = x0[:2]
        if t == 0:
            u_t = reluqp_controller.solve()
        else:
            u_t = reluqp_controller.solve(x0[:2])
        u[t, 0] = (-K @ x0[:2] + u_t)[0]
        x0[:2] = A @ x0[:2] + B_ * u[t, 0]
    print(f"ReLUQP total solve time: {reluqp_controller.solve_time}")

    # OSQP simulation
    for t in range(M):
        x[t, 2:4] = x0[2:4]
        if t == 0:
            u_t = osqp_controller.solve()
        else:
            u_t = osqp_controller.solve(x0[2:4])
        u[t, 1] = (-K @ x0[2:4] + u_t)[0]
        x0[2:4] = A @ x0[2:4] + B_ * u[t, 1]
    print(f"OSQP total solve time: {osqp_controller.solve_time}")


    # PySolver simulation
    for t in range(M):
        x[t, 4:] = x0[4:]
        if t == 0:
            u_t = pysolver_controller.solve()
        else:
            u_t = pysolver_controller.solve(x0[4:])
        u[t, 2] = (-K @ x0[4:] + u_t)[0]
        x0[4:] = A @ x0[4:] + B_ * u[t, 2]
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