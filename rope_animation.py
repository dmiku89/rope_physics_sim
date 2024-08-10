from matplotlib import pyplot as plt
from rope_model import Rope
from matplotlib import animation
import numpy as np

def animated_rope_plot(rope, *plot_args, **plot_kwargs):
    # animate sample rope
    result = rope.get_data()
    n_nodes = len(rope.nodes)
    length = len(result['time'])
    x_arr = np.vstack([[0.0]*length] + [result[f'x_{n + 1}'] for n in range(n_nodes)]).transpose()
    y_arr = np.vstack([[0.0]*length] + [result[f'y_{n + 1}'] for n in range(n_nodes)]).transpose()

    fig, ax = plt.subplots()
    ax.axis([0, 2 * np.pi, -2, 2])
    l, = ax.plot([], [], *plot_args, **plot_kwargs)
    ax.set_xlim(-1.3*n_nodes*rope.head.L0, 1.3*n_nodes*rope.head.L0)
    ax.set_ylim(-2.0*n_nodes*rope.head.L0, 1.0*n_nodes*rope.head.L0)

    def animate(i):
        l.set_data(x_arr[i], y_arr[i])

    ani = animation.FuncAnimation(fig, animate, frames=length, interval=1)
    plt.show()

    return ax, fig