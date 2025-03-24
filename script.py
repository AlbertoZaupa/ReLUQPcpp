import numpy as np
from solver_wrapper import PySolver

# Example data (must be of type np.float32 and contiguous)
nx = 3
nc = 5
rho = 0.1

H = np.array([6, 2, 1, 2, 5, 2, 1, 2, 4], dtype=np.float32)
A = np.array([1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1], dtype=np.float32)
T = np.array([0.27741242, 0.27728448, -0.27724179, -0.30141514, 0.42981273, 0.03884762,
              0.15799476, -0.12442978, 0.48464509], dtype=np.float32)
M = np.array([1e3, 1e3, 1, 1, 1], dtype=np.float32)
M_inv = np.array([1e-3, 1e-3, 1, 1, 1], dtype=np.float32)
g = np.array([-8, -3, -3], dtype=np.float32)
l = np.array([3, 0, -10, -10, -10], dtype=np.float32)
u = np.array([3, 0, np.inf, np.inf, np.inf], dtype=np.float32)
eigs = np.array([0.230738285, 288.861322, 543.016374], dtype=np.float32)

# Create a PySolver instance
solver = PySolver(rho, nx, nc, H, A, T, M, M_inv, g, l, u, eigs)

# Solve the problem
solver.setup()
solver.solve()

# Access results and solve time
print("Solve time:", solver.solve_time)
results = solver.get_results()
print("Results:", results)
