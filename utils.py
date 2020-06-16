import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

def printMatrix(a):
    """Prints the values of the input array as a matrix."""
    for row in a:
        for col in row:
            print("{:.2E}".format(col), end=" ")
        print("")

def print_modes_dataframe(data, headers, decimals):
    """Prints the columns of the "data" matrix, the first of which
    is expected to contain mode numbers, as a dataframe, with
    the precision given by "decimals".
    Column names are given by "headers".
    """
    format_str = '{:.' + str(decimals) + 'f}'
    pd.options.display.float_format = format_str.format
    df = pd.DataFrame(data=data, columns=headers)
    df.set_index(headers[0])
    df[headers[0]] = df[headers[0]].map('{:.0f}'.format)
    return df

def plot_modes(modes):
    """Plots the columns of the input array as the
    transverse modes of a building.
    """
    n_dof = modes.shape[0]
    n_modes = modes.shape[1]
    horizontal_coords = np.zeros(n_dof + 1)
    vertical_coords = np.linspace(0, n_modes, n_modes+1)
    modes_plot = np.zeros((n_dof + 1, n_modes))
    modes_plot[1:, :] = modes

    fig, ax = plt.subplots()
    ax.set(xlim=[-1,3], ylim=[0, n_modes+1])
    ax.grid()
    ax.set_aspect('equal')
    ax.plot(np.zeros(n_dof + 1), vertical_coords, color='k', marker='o')
    for col in range(n_modes):
        ax.plot(modes_plot[:,col],
                vertical_coords,
                marker='o',
                label='modo {}'.format(col+1),
               )
    plt.legend(loc='best')
    return ax

def animate_modes(
    modes,
    rot_step=0.2, nframes=10,
    interval=200, gif_name='modes_gif'
):
    """Saves gif animation of the modes given by
    the columns of the input array.
    """
    n_dof = modes.shape[0]
    n_modes = modes.shape[1]
    horizontal_coords = np.zeros(n_dof + 1)
    vertical_coords = np.linspace(0, n_modes, n_modes+1)
    modes_plot = np.zeros((n_dof + 1, n_modes))
    modes_plot[1:, :] = modes

    fig, ax = plt.subplots()
    fig.add_axes()
    ax.set(xlim=[-1,1], ylim=[0, n_modes+1])
    ax.grid()
    ax.set_aspect('equal')
    lines = np.empty(n_modes, dtype=object)
    for mode in range(n_modes):
        lines[mode], = ax.plot([], [], lw=2, marker='o')

    def init():
        for mode in range(n_modes):
            lines[mode].set_data([], [])
        return lines

    def animate(i):
        for mode in range(n_modes):
            lines[mode].set_data(modes_plot[:,mode]*np.sin(rot_step*np.pi*i),
                                 vertical_coords)
        return lines

    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=nframes, interval=interval, blit=True)
    anim.save('{}.gif'.format(gif_name), writer='imagemagick')
    return ax

def plot_modes_complexity(modes):
    """Plots the complex components of the modes matrix in
    polar coordinates. Rows and columns are read as DOFs and
    modes respectively.
    """
    total_dofs = modes.shape[0]
    total_modes = modes.shape[1]

    fix, ax = plt.subplots(1, total_modes, subplot_kw=dict(polar=True), figsize=(12,20))
    plt.tight_layout()
    for mode in range(total_modes):
        ax[mode].set_title('Modo {}'.format(mode+1), y=1.15)
        ax[mode].yaxis.set_ticks([0])
        for dof in range(total_dofs):
            ax[mode].plot([0, np.angle(modes[dof, mode])],
                     [0,np.abs(modes[dof, mode])],
                     marker='o')
    return None