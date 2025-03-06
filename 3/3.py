#b)
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sigma = 10
rho = 48
beta = 3

def lorenz(t, state):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

t_span = (0, 12)
t_eval = np.linspace(*t_span, 5000)

initial_state = [1, 1, 1]

sol = solve_ivp(lorenz, t_span, initial_state, t_eval=t_eval)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], lw=0.5)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Lorenz Attractor")
plt.savefig("HW4/3_lorenz_attractor.png", dpi=300)
plt.show()

#c)
import matplotlib.animation as animation

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')
ax.set_xlim([min(sol.y[0]), max(sol.y[0])])
ax.set_ylim([min(sol.y[1]), max(sol.y[1])])
ax.set_zlim([min(sol.y[2]), max(sol.y[2])])

line, = ax.plot([], [], [], lw=1)

def update(num):
    line.set_data(sol.y[0][:num], sol.y[1][:num])
    line.set_3d_properties(sol.y[2][:num])
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=10)

ani.save("HW4/3_lorenz.mp4", writer="ffmpeg")

plt.show()