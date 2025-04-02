import numpy as np
import matplotlib.pyplot as plt
import argparse
from solver_wrapper import CppSolver
from reluqp import ReLU_QP
from osqp import OSQP
from utils import qp_initialization
from scipy import sparse
import torch

def rk4(f, x, u, dt):
    k1 = dt*f(x, u)
    k2 = dt*f(x + k1 / 2, u)
    k3 = dt*f(x + k2 / 2, u)
    k4 = dt*f(x + k3, u)
    return x + (k1 + 2*k2 + 2*k3 + k4) / 6


class RHC_controller:

    def __init__(self, solver_class, x0, H, A, g, l, u, g_x0, lu_x0):
        self.H = H
        self.nu = self.H.shape[0]
        self.nx = x0.shape[0]
        self.A = A
        self.g = g
        self.l = l
        self.u = u
        self.g_x0 = g_x0
        self.lu_x0 = lu_x0
        self.solve_time = 0
        assert solver_class in [OSQP, ReLU_QP, CppSolver]
        self.solver_class = solver_class
        l_ = self.l + self.lu_x0 @ x0
        u_ = self.u + self.lu_x0 @ x0
        g_ = self.g + self.g_x0 @ x0
        if solver_class == OSQP:
            self.solver = OSQP()
            self.solver.setup(P=sparse.csc_matrix(self.H), q=g_, A=sparse.csc_matrix(self.A), l=l_, u=u_, eps_abs=1e-3, eps_rel=0, verbose=False)
        elif solver_class == ReLU_QP:
            self.solver = ReLU_QP()
            self.solver.setup(self.H, g_, self.A, l_, u_, precision=torch.float32)
        elif solver_class == CppSolver:
            T, eigs, M, M_inv = qp_initialization(self.H, self.E, self.l, self.upp)
            self.solver = CppSolver(0.1, self.H.shape[0], self.A.shape[0], self.H, self.A, T, M, M_inv, g_, l_, u_, eigs)
            self.solver.setup()

    def solve(self, x):
        g_ = self.g + self.g_x0 @ x
        l_ = self.l + self.lu_x0 @ x
        u_ = self.u + self.lu_x0 @ x
        if self.solver_class == OSQP:
            self.solver.update(q=g_, l=l_, u=u_)
        elif self.solver_class == ReLU_QP:
            self.solver.update(g=g_, l=l_, u=u_)
        elif self.solver_class == CppSolver:
            self.solver.update(g=g_, l=l_, u=u_, rho=-1)
        result = self.solver.solve()
        self.solve_time += result.info.solve_time
        if self.solver_class == OSQP:
            control = result.x[:self.nu]
        elif self.solver_class == ReLU_QP:
            control = result.x[:self.nu].cpu().numpy()
        elif self.solver_class == CppSolver:
            control = result.info.x_sol[:self.nu]
        return control


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run controller with a solver flag.")
    parser.add_argument("--solver", type=str, required=True, help="Solver flag value (string).")
    parser.add_argument("--plot", type=int, required=True, help="Plot flag value (int).")
    args = parser.parse_args()
    SOLVER_FLAG = args.solver  # Use the passed argument
    PLOT_FLAG = args.plot != 0
    assert SOLVER_FLAG in ["reluqp", "osqp", "cppsolver", "reluqp_warm"]

    # Loading the data
    data = np.load('atlas_dynamics.npz')
    H = data['H']
    g = data['g']
    A = data['A']
    l = data['l']
    u = data['u']
    g_x0 = data['g_x0']
    lu_x0 = data['lu_x0']

    reluqp = ReLU_QP()
    reluqp.setup(H, g, A, l, u, precision=torch.float32)
    H = np.ascontiguousarray(H)
    A = np.ascontiguousarray(A)
    g = np.ascontiguousarray(g)
    l = np.ascontiguousarray(l)
    u = np.ascontiguousarray(u)
    T, eigs, M, M_inv = qp_initialization(H, A, l, u)
    cppsolver = CppSolver(0.1, H.shape[0], A.shape[0], H, A, T, M, M_inv, g, l, u, eigs)
    cppsolver.setup()
    exit(0)

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
    du = np.ones(4)
    cu = np.ones(4)
    M = 200
    
    theta_x = np.zeros(M)
    theta_y = np.zeros(M)
    u = np.zeros((M, 4))

    # π / 6 tilt along all axes
    x0 = np.array([0, 0, 0, 0, 0, 0, np.pi/6, 0, -np.pi/6, 0, 0, 0])
    # Constant force applied to compensate gravity
    u_const = np.ones(4)
    
    solve_times = []
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
        fig, ax = plt.subplots()
        ax.plot(np.array(range(M)) * dt, theta_x, color="royalblue", linestyle="-", linewidth=2, label=r"$\theta_x(t)$")
        ax.plot(np.array(range(M)) * dt, theta_y, color="tomato", linestyle="--", linewidth=2, label=r"$\theta_y(t)$")
        ax.set_ylabel(r"$\theta(t)$")
        ax.set_xlabel('t')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"/shared/quadrotor_angles.pdf")

        fig, ax = plt.subplots()
        ax.plot(np.array(range(M)) * dt, u[:, 0], color="royalblue", linestyle="-", linewidth=2, label=r"$u_1(t)$")
        ax.plot(np.array(range(M)) * dt, u[:, 1], color="tomato", linestyle="--", linewidth=2, label=r"$u_2(t)$")
        ax.plot(np.array(range(M)) * dt, u[:, 2], color="seagreen", linestyle="-", linewidth=2, label=r"$u_3(t)$")
        ax.plot(np.array(range(M)) * dt, u[:, 3], color="purple", linestyle="--", linewidth=2, label=r"$u_4(t)$")
        ax.set_ylabel(r"$u(t)$")
        ax.set_xlabel('t')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"/shared/quadrotor_inputs.pdf")

