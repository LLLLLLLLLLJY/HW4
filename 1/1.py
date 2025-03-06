#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import os

# 创建保存图像的目录
save_dir = "HW4/1"
os.makedirs(save_dir, exist_ok=True)

# Logistic Map Function
def logistic_map(x, r):
    """Compute the logistic map function."""
    return r * x * (1 - x)

# (a) Fixed Points and Stability
def find_fixed_points(r):
    """Find the fixed points and their stability for a given r."""
    solutions = fsolve(lambda x: x - logistic_map(x, r), [0, 1])
    stability = [abs(r * (1 - 2 * x)) for x in solutions]
    return solutions, stability

# Compute fixed points for different r values
r_test_values = [1, 2, 3, 4]
for r in r_test_values:
    fixed_pts, stability = find_fixed_points(r)
    print(f"r = {r}, Fixed Points: {fixed_pts}, Stability: {stability}")

# (b) Dynamic Programming - Iterating the Logistic Map
def simulate_logistic_map(r, x0, steps=100):
    """Iterate the logistic map function for a given r and initial x0."""
    trajectory = np.zeros(steps)
    trajectory[0] = x0
    for i in range(1, steps):
        trajectory[i] = logistic_map(trajectory[i - 1], r)
    return trajectory

# Plot logistic map iteration for different r values
r_list = [2, 3, 3.5, 3.8, 4.0]
x_init = 0.2
plt.figure()
for r in r_list:
    x_series = simulate_logistic_map(r, x_init)
    plt.plot(x_series, label=f"r={r}")
plt.xlabel("Iterations")
plt.ylabel("$x_n$")
plt.title("Logistic Map Iterations for Different $r$ Values")
plt.legend()
plt.savefig(os.path.join(save_dir, "logistic_iterations.png"))
plt.close()

# (c) Different Initial Conditions
initial_x_values = [0.1, 0.3, 0.5]
r_fixed = 3.5
plt.figure()
for x0 in initial_x_values:
    x_series = simulate_logistic_map(r_fixed, x0)
    plt.plot(x_series, label=f"$x_0$={x0}")
plt.xlabel("Iterations")
plt.ylabel("$x_n$")
plt.title(f"Logistic Map for $r={r_fixed}$ with Different Initial Conditions")
plt.legend()
plt.savefig(os.path.join(save_dir, "logistic_initial_conditions.png"))
plt.close()

# (d) Bifurcation Diagram
r_values = np.linspace(2.4, 4.0, 600)
num_iterations = 1000
last_points = 200
bif_r = []
bif_x = []

for r in r_values:
    x = 0.2
    for _ in range(num_iterations):
        x = logistic_map(x, r)  # Iterate to reach steady-state
    for _ in range(last_points):  # Store last few iterations
        x = logistic_map(x, r)
        bif_r.append(r)
        bif_x.append(x)

plt.figure(figsize=(8, 6))
plt.scatter(bif_r, bif_x, s=0.1, color='black')
plt.xlabel("$r$ (Control Parameter)")
plt.ylabel("$x_n$ (Population)")
plt.title("Bifurcation Diagram of the Logistic Map")
plt.savefig(os.path.join(save_dir, "bifurcation_diagram.png"))
plt.close()

# (e) Scaling in Bifurcation
def generalized_logistic_map(x, r, gamma):
    """Compute the modified logistic map function with an additional gamma parameter."""
    return r * x * (1 - x**gamma)

gamma_vals = np.linspace(0.5, 1.5, 100)
bifurcation_pts = []

for gamma in gamma_vals:
    r_fixed = 3.0
    x = 0.2
    for _ in range(num_iterations):
        x = generalized_logistic_map(x, r_fixed, gamma)
    bifurcation_pts.append(x)

plt.figure()
plt.plot(gamma_vals, bifurcation_pts, marker='o', linestyle='-')
plt.xlabel("Gamma ($\gamma$)")
plt.ylabel("First Bifurcation Point")
plt.title("First Bifurcation Point vs. Gamma")
plt.savefig(os.path.join(save_dir, "bifurcation_vs_gamma.png"))
plt.close()