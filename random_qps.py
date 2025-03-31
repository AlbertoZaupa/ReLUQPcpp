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
        # For iterations
        self.iterations = []  # Temporary storage for each problem size (5 runs)
        self.avg_iterations = []
        self.min_iterations = []  # Store minimum value for error bar
        self.max_iterations = []  # Store maximum value for error bar

        # For solve times
        self.solve_time = []
        self.avg_solve_times = []
        self.min_solve_times = []  # Store minimum solve time for error bar
        self.max_solve_times = []  # Store maximum solve time for error bar

        self.setup_time = []
        self.avg_setup_times = []
        self.std_setup_times = []  # unchanged if needed

        self.mem_usage = []
        self.avg_mem_usage = []
        self.std_mem_usage = []  # unchanged if needed

    def compute_iterations_statistics(self):
        mean_val = np.mean(self.iterations)
        min_val = np.min(self.iterations)
        max_val = np.max(self.iterations)
        self.avg_iterations.append(mean_val)
        self.min_iterations.append(min_val)
        self.max_iterations.append(max_val)
        self.iterations = []  # clear for next problem size

    def compute_solve_time_statistics(self):
        mean_val = np.mean(self.solve_time)
        min_val = np.min(self.solve_time)
        max_val = np.max(self.solve_time)
        self.avg_solve_times.append(mean_val)
        self.min_solve_times.append(min_val)
        self.max_solve_times.append(max_val)
        self.solve_time = []  # clear for next problem size

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
    nx_list = np.hstack((nx_list[:3], nx_list))

    solver_data_list = []
    for solver in solvers:
        solver_data_list.append(SolverData())

    for j, nx in enumerate(nx_list):
        print("nx: ", int(nx))
        for seed in tqdm.tqdm(range(n_seeds)):
            for idx, solver in enumerate(solvers):
                results = solve(solver, nx=int(nx), n_eq=int(nx/4), n_ineq=int(nx/4), seed=seed, tol=tol, check_interval=solver.check_interval)
                # Only collect data for j>=3 (to skip warmup if desired)
                if j >= 3:
                    solver_data_list[idx].iterations.append(results.info.iter)
                    solver_data_list[idx].solve_time.append(results.info.solve_time)
        if j >= 3:
            for solver_data in solver_data_list:
                solver_data.compute_iterations_statistics()
                solver_data.compute_solve_time_statistics()

    # Plotting solve times using error bars from min and max
    fig, ax = plt.subplots()
    for idx, solver_data in enumerate(solver_data_list):
        # Calculate asymmetric error bars: lower = mean - min, upper = max - mean
        avg = np.array(solver_data.avg_solve_times)
        lower_error = avg - np.array(solver_data.min_solve_times)
        upper_error = np.array(solver_data.max_solve_times) - avg
        asymmetric_error = np.vstack((lower_error, upper_error))
        ax.errorbar(nx_list[3:], avg, yerr=asymmetric_error, marker='o',
                    linestyle='-', capsize=5, label=f"{solvers[idx].solver_type}")
    ax.set_xlabel('problem size')
    ax.set_ylabel('solve time (s)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"/shared/solve_time_comparison.pdf")
    
    # Plotting iterations using error bars from min and max
    fig, ax = plt.subplots()
    for idx, solver_data in enumerate(solver_data_list):
        avg = np.array(solver_data.avg_iterations)
        lower_error = avg - np.array(solver_data.min_iterations)
        upper_error = np.array(solver_data.max_iterations) - avg
        asymmetric_error = np.vstack((lower_error, upper_error))
        ax.errorbar(nx_list[3:], avg, yerr=asymmetric_error, marker='o',
                    linestyle='-', capsize=5, label=f"{solvers[idx].solver_type}")
    ax.set_xlabel('problem size')
    ax.set_ylabel('iterations')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"/shared/iterations_comparison.pdf")


if __name__ == "__main__":
    random_initial_solve([
        Solver('ReLUQP'),
        Solver('Proposed solver')],
        nx_min=10, nx_max=500, n_sample=10, n_seeds=100, tol=1e-4)
