from reluqp import ReLU_QP
from solver_wrapper import CppSolver
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import torch
import osqp
from scipy import sparse
import utils as utils


DEVICE = torch.device("cuda")
PRECISION = torch.float32
PRECISION_LABEL = "F32" if PRECISION == torch.float32 else "F64"
CHECK_INTERVAL = 25
MAX_ITER = 4000


class Solver:

    def __init__(self, solver_type, check_interval=CHECK_INTERVAL):
        assert solver_type in ['ReLUQP', 'Proposed solver', 'OSQP']
        self.solver_type = solver_type
        self.check_interval = check_interval


class SolverData:

    def __init__(self):
        self.iterations = []
        self.avg_iterations = []
        self.std_iterations = []

        self.solve_time = []
        self.avg_solve_times = []
        self.std_solve_times = []

        self.setup_time = []
        self.avg_setup_times = []
        self.std_setup_times = []

        self.mem_usage = []
        self.avg_mem_usage = []
        self.std_mem_usage = []

    def compute_iterations_statistics(self):
        self.avg_iterations.append(np.mean(self.iterations))
        self.std_iterations.append(np.std(self.iterations))
        self.iterations = []

    def compute_solve_time_statistics(self):
        self.avg_solve_times.append(np.mean(self.solve_time))
        self.std_solve_times.append(np.std(self.solve_time))
        self.solve_time = []

    def compute_setup_time_statistics(self):
        self.avg_setup_times.append(np.mean(self.setup_time))
        self.std_setup_times.append(np.std(self.setup_time))
        self.setup_time = []

    def compute_mem_usage_statistics(self):
        self.avg_mem_usage.append(np.mean(self.mem_usage))
        self.std_mem_usage.append(np.std(self.mem_usage))
        self.mem_usage = []


def solve(solver, nx=10, n_eq=5, n_ineq=5, seed=1, tol=1e-4, check_interval=CHECK_INTERVAL):
    H, g, A, l, u, x_sol = utils.rand_qp(nx=nx, n_eq=n_eq, n_ineq=n_ineq, seed=seed)
    
    model = None
    if solver.solver_type == 'ReLUQP':
        model = ReLU_QP()
        model.setup(H=H, g=g, A=A, l=l, u=u, eps_abs=tol, device=DEVICE, precision=PRECISION, check_interval=check_interval, max_iter=MAX_ITER)
    elif solver.solver_type == 'OSQP':
        model = osqp.OSQP()
        model.setup(P=sparse.csc_matrix(H), q=g, A=sparse.csc_matrix(A), l=l, u=u, eps_abs=1e-3, eps_rel=0, verbose=False)
    elif solver.solver_type == 'Proposed solver':
        T, eigs, M, M_inv = utils.qp_initialization(H, A, l, u)
        model = CppSolver(0.1, H.shape[0], A.shape[0], H, A, T, M, M_inv, g, l, u, eigs, reactive_rho_duration=5)
    
    results = model.solve()

    return results


def random_initial_solve(solvers, nx_min=10, nx_max=1000, n_sample=10, n_seeds=5, tol=1e-4):
    nx_list = np.linspace(nx_min, nx_max, num=n_sample)
        
    solver_data_list = []
    for solver in solvers:
        solver_data_list.append(SolverData())

    for nx in nx_list:
        print("nx: ", int(nx))
        for seed in tqdm.tqdm(range(n_seeds)):
            for idx, solver in enumerate(solvers):
                results = solve(solver, nx=int(nx), n_eq=int(nx/4), n_ineq=int(nx/4), seed=seed, tol=tol, check_interval=solver.check_interval)
                #assert results.info.iter < MAX_ITER, idx
                solver_data_list[idx].iterations.append(results.info.iter)
                solver_data_list[idx].solve_time.append(results.info.solve_time)

        for solver_data in solver_data_list:
            solver_data.compute_iterations_statistics()
            solver_data.compute_solve_time_statistics()

    plt.style.use("ggplot")
    
    # solve time plots
    fig, ax = plt.subplots()
    for idx, solver_data in enumerate(solver_data_list):
        ax.errorbar(nx_list, solver_data.avg_solve_times, yerr=solver_data.std_solve_times, marker='o',
                linestyle='-', capsize=5, label=f"{solvers[idx].solver_type}")
    ax.set_xlabel('problem size')
    ax.set_ylabel('solve time (s)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"/shared/solve_time_comparison.png")
    
    # iterations plots
    fig, ax = plt.subplots()
    for idx, solver_data in enumerate(solver_data_list):
        ax.errorbar(nx_list, solver_data.avg_iterations, yerr=solver_data.std_iterations, marker='o', linestyle='-', capsize=5, label=f"{solvers[idx].solver_type}")
    ax.set_xlabel('problem size')
    ax.set_ylabel('iterations')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"/shared/iterations_comparison.png")


if __name__ == "__main__":
    random_initial_solve([
        Solver('ReLUQP'),
        Solver('Proposed solver')],
        nx_min=10, nx_max=500, n_sample=15, n_seeds=5, tol=1e-4)