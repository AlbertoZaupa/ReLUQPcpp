import numpy as np
import matplotlib.pyplot as plt
import subprocess
import re

def run_solver(solver, runs=10, extra_args=None):
    """
    Runs the solver command repeatedly, parses output, and returns lists of average and worst-case times.
    :param solver: The solver identifier to pass to the command (e.g. 'cppsolver', 'osqp', or 'reluqp')
    :param runs: Number of runs
    :param extra_args: List of extra arguments to append to the command (e.g. ['--warm_start'])
    :return: (avg_times, worst_times) lists of floats
    """
    avg_times = []
    worst_times = []
    base_command = ['python3', 'examples/quadrotor_stabilization.py', '--solver', solver, '--plot', '0']
    if extra_args:
        base_command.extend(extra_args)
    for _ in range(runs):
        # Run the command and decode the output as a string
        output = subprocess.check_output(base_command, universal_newlines=True)
        
        # Parse average solve time
        avg_match = re.search(r"average solve time:\s+(\S+)", output)
        worst_match = re.search(r"worst case time:\s+(\S+)", output)
        if avg_match and worst_match:
            avg_time = float(avg_match.group(1))
            worst_time = float(worst_match.group(1))
            avg_times.append(avg_time)
            worst_times.append(worst_time)
        else:
            raise ValueError("Could not parse output:\n" + output)
    return avg_times, worst_times

# Define the solvers with labels and the corresponding command line arguments
solvers = [
    ("Proposed solver", "cppsolver"),
    ("OSQP", "osqp"),
    ("ReLUQP", "reluqp_warm")
]

# Run experiments for each solver
avg_solve_time = []
worst_case_time = []
for label, solver_arg in solvers:
    avg, worst = run_solver(solver_arg, runs=10)
    print(f"ran solver: {label}")
    avg_solve_time.append(avg)
    worst_case_time.append(worst)

# Compute mean and standard deviation for each algorithm
means = [np.mean(times) for times in avg_solve_time]
std_devs = [np.std(times, ddof=1) for times in avg_solve_time]

# Create bar chart with error bars for average solve times
fig, ax = plt.subplots()
ax.bar([s[0] for s in solvers], means, yerr=std_devs, capsize=10,
       color=['blue', 'lightgreen', 'orange'], edgecolor='black')
ax.set_xlabel("Solvers")
ax.set_ylabel("Average solve time (s)")
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("./results/quadrotor_avg_solve_times.pdf")
plt.close()

means_worst = [np.mean(times) for times in worst_case_time]
std_devs_worst = [np.std(times, ddof=1) for times in worst_case_time]
f = open("/shared/quadrotor_stabilization.txt", "w")
f.write(f"cppsolver average solve time: {means[0]} +- {std_devs[0]}\n")
f.write(f"cppsolver worst case time: {means_worst[0]} +- {std_devs_worst[0]}\n")
f.write(f"osqp average solve time: {means[1]} +- {std_devs[1]}\n")
f.write(f"osqp worst case time: {means_worst[1]} +- {std_devs_worst[1]}\n")
f.write(f"reluqp average solve time: {means[2]} +- {std_devs[2]}\n")
f.write(f"reluqp worst case time: {means_worst[2]} +- {std_devs_worst[2]}\n")
