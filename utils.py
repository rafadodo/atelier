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

def plot_column_modes(modes):
    """Plots the columns of the input array as the
    transverse modes of a column-like structure.
    """
    n_dof = modes.shape[0]
    n_modes = modes.shape[1]
    vertical_coords = np.linspace(0, n_dof, n_dof+1)
    modes_plot = np.zeros((n_dof + 1, n_modes))
    for mode in range(n_modes):
        modes_plot[1:, mode] = np.real(modes[:,mode]) / max(abs(modes[:,mode]))

    fig, ax = plt.subplots()
    ax.set_title('Formas Modales')
    ax.set(xlim=[-1, 3], ylim=[0, n_dof+1])
    ax.set_ylabel('DOFs')
    ax.set_xticklabels([])
    n_ticks = len(ax.get_yticks())
    if n_ticks>(n_dof+2):
        ax.set_yticks(ax.get_yticks()[::2][1:])
    ax.grid()
    ax.set_aspect('equal')

    ax.plot(np.zeros(n_dof + 1), vertical_coords, color='k', marker='o')
    for col in range(n_modes):
        ax.plot(modes_plot[:,col],
                vertical_coords,
                marker='o',
                label='modo {}'.format(col+1),
               )
    plt.legend(loc='upper right')
    plt.show()
    return None

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

def plot_3d_mode(node_order, coords, connections, mode_vec, magnif):
    """"""
    undeformed = np.zeros((len(coords), 3))
    for node_idx in range(len(coords)):
        node_name = list(node_order.keys())[list(node_order.values()).index(node_idx)]
        undeformed[node_idx,:] = coords[node_name]
    delta_pos = np.real(mode_vec.reshape((len(coords)-1, 3)))/max(np.abs(mode_vec))
    deformed = undeformed + magnif*(np.concatenate((np.zeros((1,3)), delta_pos)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(undeformed[:,0],undeformed[:,1],undeformed[:,2], color='k')
    ax.scatter(deformed[:,0],deformed[:,1],deformed[:,2], color='r')
    for source, targets in connections.items():
        for target in targets:
            i, j = node_order[source], node_order[target]
            ax.plot([undeformed[i,0], undeformed[j,0]],
                    [undeformed[i,1], undeformed[j,1]],
                    [undeformed[i,2], undeformed[j,2]],
                    color='k', linewidth=0.75)
            ax.plot([deformed[i,0], deformed[j,0]],
                    [deformed[i,1], deformed[j,1]],
                    [deformed[i,2], deformed[j,2]],
                    color='r', linewidth=0.75)
    return None