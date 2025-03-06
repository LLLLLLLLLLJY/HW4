#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import ConvexHull

# 创建保存目录
save_dir = "HW4/2"
os.makedirs(save_dir, exist_ok=True)

# (a) Julia Set Generation
def julia_set(c, width=800, height=800, x_min=-1.5, x_max=1.5, y_min=-1, y_max=1, max_iter=256):
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    C = np.full(Z.shape, c)
    
    iter_count = np.zeros(Z.shape, dtype=int)
    mask = np.ones(Z.shape, dtype=bool)
    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + C[mask]
        mask &= (np.abs(Z) < 2)
        iter_count[mask] = i
    
    return X, Y, iter_count

c = -0.7 + 0.356j
X, Y, iter_count = julia_set(c)

plt.figure(figsize=(8, 8))
plt.imshow(iter_count, extent=(-1.5, 1.5, -1, 1), cmap='inferno')
plt.colorbar()
plt.title(f"Julia Set with c={c}")
plt.xlabel("Real")
plt.ylabel("Imaginary")
plt.savefig(f"{save_dir}/julia_set.png")  # 保存 Julia 集合图像
plt.show()

# (b) Convex Hull Area
def convex_hull_area(X, Y, iter_count):
    points = np.c_[X[iter_count > 0].flatten(), Y[iter_count > 0].flatten()]
    hull = ConvexHull(points)
    return hull.volume

hull_area = convex_hull_area(X, Y, iter_count)
print(f"Convex Hull Area: {hull_area}")

# (c) Contour Area
def contour_area(X, Y, iter_count, level=100):
    plt.figure(figsize=(8, 8))
    plt.contour(X, Y, iter_count, levels=[level], colors='red')
    plt.imshow(iter_count, extent=(-1.5, 1.5, -1, 1), cmap='inferno')
    plt.colorbar()
    plt.title(f"Julia Set Contour with c={c}")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.savefig(f"{save_dir}/julia_set_contour.png")  # 保存轮廓图
    plt.show()

contour_area(X, Y, iter_count)

# (d) Box-Counting Method for Fractal Dimension
def box_counting(X, Y, iter_count):
    sizes = np.logspace(1, 5, num=10, base=2, dtype=int)
    counts = []
    for size in sizes:
        grid = np.zeros((size, size))
        for i in range(iter_count.shape[0]):
            for j in range(iter_count.shape[1]):
                if iter_count[i, j] > 0:
                    grid[i * size // iter_count.shape[0], j * size // iter_count.shape[1]] = 1
        counts.append(np.sum(grid))
    
    sizes = 1 / sizes
    log_counts = np.log(counts)
    log_sizes = np.log(sizes)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    return coeffs[0]

fractal_dim = box_counting(X, Y, iter_count)
print(f"Fractal Dimension: {fractal_dim}")

# 保存结果到文本文件
with open(f"{save_dir}/results.txt", "w") as f:
    f.write(f"Convex Hull Area: {hull_area}\n")
    f.write(f"Fractal Dimension: {fractal_dim}\n")