import numpy as np
import torch
from reluqp import ReLU_QP
import matplotlib.pyplot as plt

class RHC_controller():
    
    def __init__(self, x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf):
        self.K = K
        self.condense(A, B, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf)
        self.g = (self.M @ x0).reshape(-1)
        const = np.zeros(self.N * (self.nu_c + self.nx_c))
        const[:self.N * self.nu_c] = self.F1 @ x0
        const[self.N * self.nu_c:] = self.F2 @ x0
        self.upp = self.d - const
        self.l = self.c - const

        self.solve_time = 0
        self.worst_case_time = 0
        self.worst_case_time_iter = 0

    def condense(self, A, B, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf):
        # Dimensions
        nx = A.shape[1]
        self.nx = nx
        nu = B.shape[1]
        self.nu = nu
        self.N = N
        nu_c = du.shape[0]
        nx_c = dx.shape[0]
        self.nx_c = nx_c
        self.nu_c = nu_c

        Q_bar = np.kron(np.eye(N), Q)
        Q_bar[-nx:, -nx:] = Pf
        R_bar = np.kron(np.eye(N), R)
        
        G = np.zeros((nx * N, nu * N))
        G_bar = np.eye(N * nu)
        A_kB = B
        KA_kB = self.K @ B
        for k in range(N):
            for i in range(k, N):
                G[i * nx:(i + 1) * nx, (i - k) * nu:(i - k + 1) * nu] = A_kB
            for i in range(k + 1, N):
                G_bar[i * nu:(i + 1) * nu, (i - k - 1) * nu:(i - k) * nu] = - KA_kB
            A_kB = (A - B @ self.K) @ A_kB
            KA_kB = self.K @ A_kB

        self.H = R_bar + G.T @ Q_bar @ G

        L = np.zeros((nx * N, nx))
        L_bar = np.zeros((nu * N, nx))
        Ak = np.eye(nx)
        for k in range(N):
            L_bar[k * nu:(k + 1) * nu, :] = - self.K @ Ak
            Ak = (A - B @ self.K) @ Ak
            L[k * nx:(k + 1) * nx, :] =  Ak
        self.M = G.T @ Q_bar @ L

        E1_ = np.kron(np.eye(N), Eu)
        E1 = E1_ @ G_bar
        self.F1 = E1_ @ L_bar
        d1 = np.kron(np.ones((N, 1)), du).reshape(-1)
        c1 = np.kron(np.ones((N, 1)), cu).reshape(-1)

        E2_ = np.kron(np.eye(N), Ex)
        E2_[-nx_c:, -nx:] = Ef
        E2 = E2_ @ G
        self.F2 = E2_ @ L
        d2 = np.kron(np.ones((N, 1)), dx).reshape(-1)
        d2[-self.nx_c:] = df
        c2 = np.kron(np.ones((N, 1)), cx).reshape(-1)
        c2[-self.nx_c:] = cf

        self.E = np.vstack((E1, E2))
        d = np.zeros((d1.shape[0] + d2.shape[0]))
        d[:d1.shape[0]] = d1
        d[d1.shape[0]:] = d2
        self.d = d
        c = np.zeros((c1.shape[0] + c2.shape[0]))
        c[:c1.shape[0]] = c1
        c[c1.shape[0]:] = c2
        self.c = c


class ReLUQP_controller(RHC_controller):

    def __init__(self, x0, rho, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf):
        super().__init__(x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf)
        self.solver = ReLU_QP()
        self.solver.setup(self.H, self.g, self.E, self.l, self.upp, precision=torch.float64, adaptive_rho=False, rho=rho)

    def solve(self):
        result = self.solver.solve()
        return result


if __name__ == '__main__':
    g = 9.81
    m = 1.5
    Jx = Jy = 0.03
    Jz = 0.06
    l = 0.225
    c = 0.015
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
    x0 = np.array([0, 0, 0, 0, 0, 0, np.pi/6, 0, -np.pi/6, 0, 0, 0])
    rho_values = np.logspace(np.log10(0.01), np.log10(10000), num=1000)

    iters = []
    for rho in rho_values:
        controller = ReLUQP_controller(x0, rho, A, B, K, Q, R, Pf, N, Ex,
                                            dx, cx, Eu, du, cu, Ef, df, cf)
        iters.append(controller.solve().info.iter)
        
    fig, ax = plt.subplots()
    ax.plot(rho_values, iters, color="royalblue", linestyle="-", linewidth=2)
    ax.set_ylabel('Iterations until convergence')
    ax.set_xlabel(r"$\rho$")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"/shared/constant_rho.pdf")