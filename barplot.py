import numpy as np
import matplotlib.pyplot as plt

# Sample data: Execution times for 3 algorithms over 10 runs
algorithms = ["Proposed solver", "OSQP", "ReLUQP"]
avg_solve_time = [
    [0.0014130681921960787,
        0.0013737140694865958,
        0.0014220187446335332,
        0.0014405099855503067,
        0.001394032405805774,
        0.0014318430080311373,
        0.0014368569548241794,
        0.001384915960370563,
        0.0014370628370670602,
        0.0014219475234858692],  # Algorithm A
    [0.0021978458,
        0.0021853590650000015,
        0.0021800058649999996,
        0.002274301880000001,
        0.00220922224,
        0.0021957219699999995,
        0.002213422795000001,
        0.0021970773499999994,
        0.0022143180250000006,
        0.0021869001650000014],  # Algorithm B
    [0.0032584322567284084,
        0.0031627589887380588,
        0.003134978634938597,
        0.0031230211591720576,
        0.0031978427948057653,
        0.003195831050053236,
        0.0031689395998418346,
        0.0031255940743535764,
        0.003142027842774986,
        0.003227102285847069]   # Algorithm C
]

worst_case_time = [
    [0.0038984250277280807,
        0.003703170921653509,
        0.003892351873219013,
        0.003890901105478406,
        0.0038945251144468784,
        0.003904371988028288,
        0.0037033231928944588,
        0.003959940746426582,
        0.004021911881864071,
        0.00392912607640028],
    [0.007892749,
        0.007909691,
        0.007259345,
        0.007925993,
        0.00789877,
        0.007914963,
        0.007892243,
        0.007886169,
        0.007912206,
        0.007902876],
    [0.3740073166042566,
        0.35803752441704273,
        0.355862413674593,
        0.35516026651859284,
        0.3646019065231085,
        0.3636987883895636,
        0.35709553940594196,
        0.3549061743468046,
        0.3571887397766113,
        0.36964416632056235]
]

# Compute mean and standard deviation for each algorithm
means = [np.mean(times) for times in avg_solve_time]
std_devs = [np.std(times, ddof=1) for times in avg_solve_time]  # ddof=1 for sample std dev

# Create bar chart with error bars
fig, ax = plt.subplots()
ax.bar(algorithms, means, yerr=std_devs, capsize=10, color=['skyblue', 'salmon', 'lightgreen'], edgecolor='black')

# Labels and title
ax.set_xlabel("Solvers")
ax.set_ylabel("Average solve time")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.savefig(f"./results/quadrotor_avg_solve_times.png")


# Compute mean and standard deviation for each algorithm
means = [np.mean(times) for times in worst_case_time]
std_devs = [np.std(times, ddof=1) for times in worst_case_time]  # ddof=1 for sample std dev

# Create bar chart with error bars
fig, ax = plt.subplots()
ax.bar(algorithms, means, yerr=std_devs, capsize=10, color=['skyblue', 'salmon', 'lightgreen'], edgecolor='black')

# Labels and title
ax.set_xlabel("Solvers")
ax.set_ylabel("Worst case solve time")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.savefig(f"./results/quadrotor_worst_case_times.png")
