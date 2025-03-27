import numpy as np
import matplotlib.pyplot as plt

# Sample data: Execution times for 3 algorithms over 10 runs
algorithms = ["Proposed solver", "OSQP", "ReLUQP"]
avg_solve_time = [
    [0.00036629832451581026,
        0.00036800009474973193,
        0.0003630245059321169,
        0.00040510330931283536,
        0.0003908752993447706,
        0.00039873849498690107,
        0.0003941728740755934,
        0.0003953022802306805,
        0.0003766725469904486,
        0.0003671367508650292],  # Algorithm A
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
    [0.0029147618915885687,
        0.002976727904751897,
        0.002949122106656432,
        0.0030371209140866995,
        0.0029923170804977417,
        0.003148773917928338,
        0.003044015960767865,
        0.0031353430822491646,
        0.0028843150939792395,
        0.00292481598444283],
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
ax.set_ylabel("Average solve time (s)")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.savefig(f"./results/quadrotor_avg_solve_times.pdf")


# AVERAGE SOLVE TIMES, WITH RELUQP WARM STARTED

avg_solve_time[2] = [
    0.0013612968756924904,
        0.0013671646269422094,
        0.0013758507912332693,
        0.0013811587713771136,
        0.0013734898549977242,
        0.0013551827804676865,
        0.0013617662693837175,
        0.0013633960307243495,
        0.0013831183022290662,
        0.0013654567260089202
]

# Compute mean and standard deviation for each algorithm
means = [np.mean(times) for times in avg_solve_time]
std_devs = [np.std(times, ddof=1) for times in avg_solve_time]  # ddof=1 for sample std dev

# Create bar chart with error bars
fig, ax = plt.subplots()
ax.bar(algorithms, means, yerr=std_devs, capsize=10, color=['skyblue', 'salmon', 'lightgreen'], edgecolor='black')

# Labels and title
ax.set_xlabel("Solvers")
ax.set_ylabel("Average solve time (s)")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.savefig(f"./results/quadrotor_avg_solve_times_reluqp_warm.pdf")


# WORST CASE SOLVE TIMES


# Compute mean and standard deviation for each algorithm
means = [np.mean(times) for times in worst_case_time]
std_devs = [np.std(times, ddof=1) for times in worst_case_time]  # ddof=1 for sample std dev
print(means)
# Create bar chart with error bars
fig, ax = plt.subplots()
ax.bar(algorithms, means, yerr=std_devs, capsize=10, color=['skyblue', 'salmon', 'lightgreen'], edgecolor='black')

# Labels and title
ax.set_xlabel("Solvers")
ax.set_ylabel("Worst case solve time (s)")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.savefig(f"./results/quadrotor_worst_case_times.pdf")


# WORST CASE SOLVE TIME RELUQP EXCLUDED


# Create bar chart with error bars
fig, ax = plt.subplots()
ax.bar(algorithms[:-1], means[:-1], yerr=std_devs[:-1], capsize=10, color=['skyblue', 'salmon', 'lightgreen'], edgecolor='black')

# Labels and title
ax.set_xlabel("Solvers")
ax.set_ylabel("Worst case solve time (s)")
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.savefig(f"./results/quadrotor_worst_case_times_reluqp_excluded.pdf")
