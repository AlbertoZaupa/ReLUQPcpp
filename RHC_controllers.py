import numpy as np
import torch
import scipy
import scipy.sparse as sparse
from reluqp import ReLU_QP
from solver_wrapper import CppSolver
import osqp
from scipy.linalg import eigh

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

    def __init__(self, x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf):
        super().__init__(x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf)
        self.solver = ReLU_QP()
        self.solver.setup(self.H, self.g, self.E, self.l, self.upp, precision=torch.float32)

    def solve(self, x0=None):
        if not x0 is None:
            self.g = (self.M @ x0).reshape(-1)
            const = np.zeros(self.N * (self.nu_c + self.nx_c))
            const[:self.N * self.nu_c] = self.F1 @ x0
            const[self.N * self.nu_c:] = self.F2 @ x0
            self.upp = self.d - const
            self.l = self.c - const
        self.solver.update(g=self.g, l=self.l, u=self.upp)
        result = self.solver.solve()
        #print(f"ReLUQP solver iterations: {result.info.iter}")
        self.solve_time += result.info.solve_time
        return result.x[:self.nu].cpu().numpy()


class OSQP_controller(RHC_controller):
    def __init__(self, x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf):
        super().__init__(x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf)
        self.solver = osqp.OSQP()
        self.solver.setup(P=sparse.csc_matrix(self.H), q=self.g, A=sparse.csc_matrix(self.E), l=self.l, u=self.upp, eps_abs=1e-3, eps_rel=0, verbose=False)

    def solve(self, x0=None):
        if not x0 is None:
            self.g = (self.M @ x0).reshape(-1)
            const = np.zeros(self.N * (self.nu_c + self.nx_c))
            const[:self.N * self.nu_c] = self.F1 @ x0
            const[self.N * self.nu_c:] = self.F2 @ x0
            self.upp = self.d - const
            self.l = self.c - const
        self.solver.update(q=self.g, l=self.l, u=self.upp)
        result = self.solver.solve()
        #print(f"OSQP solver iterations: {result.info.iter}")
        self.solve_time += result.info.solve_time
        return result.x[:self.nu]


class CppSolver_controller(RHC_controller):
    def __init__(self, x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf):
        super().__init__(x0, A, B, K, Q, R, Pf, N, Ex, dx, cx, Eu, du, cu, Ef, df, cf)
        T, eigs, M, M_inv = qp_initialization(self.H, self.E, self.l, self.upp)
        self.solver = CppSolver(0.1, self.H.shape[0], self.E.shape[0], self.H, self.E, T, M, M_inv, self.g, self.l, self.upp, eigs)
        self.solver.setup()

    def solve(self, x0=None):
        if not x0 is None:
            self.g = (self.M @ x0).reshape(-1)
            const = np.zeros(self.N * (self.nu_c + self.nx_c))
            const[:self.N * self.nu_c] = self.F1 @ x0
            const[self.N * self.nu_c:] = self.F2 @ x0
            self.upp = self.d - const
            self.l = self.c - const
        if x0 is None:
            self.solver.update(g=self.g, l=self.l, u=self.upp, rho=-1)
        else:
            self.solver.update(g=self.g, l=self.l, u=self.upp, rho=0.1)
        self.solver.solve()
        result = self.solver.get_results()
        #print(f"PySolver solver iterations: {result.iter}")
        self.solve_time += result.solve_time
        return result.x_sol[:self.nu]




def qp_initialization(H: np.ndarray, A: np.ndarray, l: np.ndarray, u: np.ndarray):
    eq_tol = 1e-6
    nx = H.shape[0]
    nc = A.shape[0]

    # Decomposition for fast update of ADMM hessian
    eigs, U = eigh(H, driver="evd")
    e_inv = 1 / eigs
    e_sqrt = np.sqrt(e_inv)
    L_inv = U @ np.diag(e_sqrt) @ U.T
    H_inv = U @ np.diag(e_inv) @ U.T

    # Scaling matrix enforces higher penalty to equality constraints
    sqrtm = np.ones(nc)
    m = np.ones(nc)
    m_inv  = np.ones(nc)
    sqrtm[(u - l) <= eq_tol] = np.sqrt(1e3)
    m[(u - l) <= eq_tol] = 1e3
    m_inv[(u - l) <= eq_tol] = 1e-3
    sqrtM = np.diag(sqrtm)

    S = L_inv @ A.T @ sqrtM
    eigs, U = eigh(S @ S.T, driver="evd")
    T = L_inv.T @ U
    return T, eigs, m, m_inv