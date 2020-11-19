import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure(figsize=(6, 6))
ax = plt.gca()
ax.grid()
ln1, = ax.plot([], [], '-', lw=2)
ln2, = ax.plot([], [], '-', color='r', lw=2)
theta = np.linspace(0, 2*np.pi, 100)
r_out = 1
r_in = 0.5

def init():
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    x_out = [r_out*np.cos(theta[i]) for i in range(len(theta))]
    y_out = [r_out*np.sin(theta[i]) for i in range(len(theta))]
    ln1.set_data(x_out, y_out)
    return ln1,

def update(i):
    x_in = [(r_out-r_in)*np.cos(theta[i])+r_in*np.cos(theta[j]) for j in range(len(theta))]
    y_in = [(r_out-r_in)*np.sin(theta[i])+r_in*np.sin(theta[j]) for j in range(len(theta))]
    ln2.set_data(x_in, y_in)
    return ln2,

ani = animation.FuncAnimation(fig, update, range(len(theta)), init_func=init, interval=30)
#ani.save('roll.gif', writer='imagemagick', fps=100)

plt.show()