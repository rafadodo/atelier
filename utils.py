# -*- coding: utf-8 -*-
"""
This module contains helper functions for analyzing and visualizing structural dynamics data.

Functions:
- print_matrix(mat): Prints the values of the input array in 2D-matrix fashion.
- get_max_off_diagonal(mat): Obtains the maximum value of an array ignoring its diagonal.
- print_modes_dataframe(modal_data, headers, decimals, save_title): Prints the input matrix as a
dataframe and saves it in a LaTeX file.
- plot_column_modes(mode_shapes): Plots the columns of the input array as the transverse mode
shapes of a column-like structure.
- animate_modes(mode_shapes, rot_step, nframes, interval, gif_name): Saves a GIF animation of the
mode shapes given by the columns of the input array.
- plot_modes_complexity(mode_shapes): Plots the complex components of the input matrix in polar
coordinates.
- plot_3d_mode(node_order, coords, connections, mode_vec, mode_name='', magnif=1): Plots a 3D
representation of a mode shape.
- get_damp_from_decay(decay): Estimates the damping of a decaying time-domain signal.
- get_freq_from_signal(timestamps, values): Estimates the frequency of a time-domain signal by
averaging the times between zero crossings.
"""
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def print_matrix(mat):
    """Prints the rows of the input (square) array.

    Parameters:
        mat (numpy.ndarray, shape (n_rows, n_cols)):
            The input matrix to be printed.
            - n_rows: Number of rows in the matrix.
            - n_cols: Number of columns in the matrix.

    Raises:
        ValueError: If the input array is not 2D.
    """
    if len(mat.shape) != 2:
        raise ValueError("Input array must be 2D. Shape provided: {}".format(mat.shape))

    for row in mat:
        for value in row:
            print("{:.2E}".format(value), end=" ")
        print("")

    return None


def get_max_off_diagonal(mat):
    """Obtains the maximum value of a (square) array ignoring its diagonal.

    Parameters:
        mat (numpy.ndarray, shape (n, n)):
            The input matrix.
            - n: Number of rows and columns in the square matrix.

    Returns:
        float: The maximum value off the diagonal.

    Raises:
        ValueError: If the input array is not square.
    """
    if mat.shape[0] != mat.shape[1]:
        raise ValueError("Input array must be square. Shape provided: {}".format(mat.shape))

    mask = np.ones(mat.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_off_diagonal = mat[mask].max()

    return max_off_diagonal


def print_modes_dataframe(modal_data, headers, decimals):
    """Prints the input modal-data matrix in pandas-dataframe format. The first column is expected
    to contain the mode numbers, and the rest of them can contain any modal characteristic, such
    as frequency or damping.

    Parameters:
        modal_data (numpy.ndarray, shape (n_modes, n_columns)):
            The input data matrix.
            - n_modes: Number of modes.
            - n_columns: Number of columns, that is, modal characteristics.
        headers (list): List of column names.
        decimals (int): Number of decimal places for data formatting.

    Returns:
        pandas.DataFrame: The formatted DataFrame.
    """
    format_str = '{:.' + str(decimals) + 'f}'
    pd.options.display.float_format = format_str.format
    df = pd.DataFrame(data=modal_data, columns=headers)
    df[headers[0]] = df[headers[0]].map('{:.0f}'.format)
    df.set_index(headers[0], inplace=True)
    df.columns.name = df.index.name
    df.index.name = None

    return df


def plot_column_modes(mode_shapes):
    """Plots the columns of the input array as the transverse displacements of a column-like
    structure.

    Parameters:
        mode_shapes (numpy.ndarray, shape (n_dof, n_modes)):
            The input mode-shape matrix.
            - n_dof: Number of degrees of freedom.
            - n_modes: Number of modes.
    """
    n_dof = mode_shapes.shape[0]
    n_modes = mode_shapes.shape[1]
    vertical_coords = np.linspace(0, n_dof, n_dof+1)
    modes_plot = np.zeros((n_dof + 1, n_modes))
    for mode in range(n_modes):
        modes_plot[1:, mode] = np.real(mode_shapes[:,mode]) / max(abs(mode_shapes[:,mode]))

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


def animate_modes(mode_shapes, rot_step=0.2, nframes=10, interval=200, gif_name='modes_gif'):
    """Saves gif animation of the mode shapes given by the columns of the input array.

    Parameters:
        mode_shapes (numpy.ndarray, shape (n_dof, n_modes)):
            The input mode-shape matrix.
            - n_dof: Number of degrees of freedom (DOFs).
            - n_modes: Number of modes.
        rot_step (float): Rotation step for animation.
        nframes (int): Number of frames for animation.
        interval (int): Interval between frames in milliseconds.
        gif_name (str): Name of the output GIF file.

    Returns:
        matplotlib.axes.Axes: The axes object containing the animation plot.
    """
    n_dof = mode_shapes.shape[0]
    n_modes = mode_shapes.shape[1]
    vertical_coords = np.linspace(0, n_modes, n_modes+1)
    modes_plot = np.zeros((n_dof + 1, n_modes))
    modes_plot[1:, :] = mode_shapes

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


def plot_modes_complexity(mode_shapes):
    """Plots the complex components of a mode-shape matrix in polar coordinates.

    Parameters:
        mode_shapes (numpy.ndarray, shape (n_dof, n_modes)):
            The input mode-shape matrix.
            - n_dof: Number of degrees of freedom.
            - n_modes: Number of modes.
    """
    total_dofs = mode_shapes.shape[0]
    total_modes = mode_shapes.shape[1]

    fix, ax = plt.subplots(1, total_modes, subplot_kw=dict(polar=True), figsize=(12,20))
    plt.tight_layout()
    for mode in range(total_modes):
        ax[mode].set_title('Modo {}'.format(mode+1), y=1.15)
        ax[mode].yaxis.set_ticks([0])
        for dof in range(total_dofs):
            ax[mode].plot([0, np.angle(mode_shapes[dof, mode])],
                        [0, np.abs(mode_shapes[dof, mode])],
                        marker='o',
                        )

    return None


def plot_3d_mode(node_order, coords, connections, mode_vec, mode_name='', magnif=1):
    """Plots a 3D representation of a mode shape.

    Parameters:
        node_order (dict):
            Dictionary mapping node labels to indices.
            - Keys (str): Node labels.
            - Values (int): Node indices.
        coords (dict):
            Dictionary of node coordinates.
            - Keys (str): Node labels.
            - Values (numpy.ndarray, shape (3,)): Coordinates (x, y, z) of the node.
        connections (dict):
            Dictionary of node connections.
            - Keys (str): Source node labels.
            - Values (list of str): Target node labels connected to the source node.
        mode_vec (numpy.ndarray): The mode shape vector.
        mode_name (str): Name of the mode.
        magnif (float): Magnification factor for the mode shape.
    """
    undeformed = np.zeros((len(coords), 3))
    for node_idx in range(len(coords)):
        node_name = list(node_order.keys())[list(node_order.values()).index(node_idx)]
        undeformed[node_idx,:] = coords[node_name]
    delta_pos = np.real(mode_vec.reshape((len(coords)-1, 3)))/max(np.abs(mode_vec))
    deformed = undeformed + magnif*(np.concatenate((np.zeros((1,3)), delta_pos)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(mode_name)
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


def get_damp_from_decay(decay):
    """Estimates the damping of a decaying time-domain signal by analyzing peak values and
    applying a linear regression to logarithmic peak ratios.

    Parameters:
        decay (numpy.ndarray, shape (n_samples,)): Time-series data representing the decaying
            response.

    Returns:
        tuple:
            - damp (float): Estimated damping ratio.
            - R2 (float): Coefficient of determination (fit quality).
            - A (numpy.ndarray): Design matrix used in the least squares regression.
            - b (numpy.ndarray): Logarithmic peak ratios.
            - c (float): Intercept from least squares fit.
            - m (float): Slope from least squares fit.
    """
    peak_ind = np.array([m for m in signal.argrelmax(abs(decay), order=1)]).flatten()
    log_peak_ratios = np.zeros(len(peak_ind)-1)
    for i in range(len(peak_ind)-1):
        log_peak_ratios[i] = 2*np.log(abs(decay[peak_ind[0]] / decay[peak_ind[i+1]]))
    peak_nums = np.linspace(1, len(log_peak_ratios), len(log_peak_ratios))

    A = np.vstack([peak_nums, np.ones(len(peak_nums))]).T
    b = log_peak_ratios

    m, c = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = np.linalg.lstsq(A, b, rcond=None)[1][0]
    R2 = 1 - resid / (b.size * b.var())

    damp = m/np.sqrt(m**2 + 4*np.pi**2)

    return damp, R2, A, b, c, m


def get_freq_from_signal(timestamps, values):
    """Estimates the frequency of a time-domain signal by averaging the times between zero
    crossings.

    Parameters:
        timestamps (numpy.ndarray, shape (n_samples,)): Time values corresponding to the signal.
        values (numpy.ndarray, shape (n_samples,)): Signal values (real part considered for zero
            crossings).

    Returns:
        tuple:
            - freq (float): Estimated signal frequency in Hz.
            - R2 (float): Coefficient of determination (fit quality).
            - A (numpy.ndarray): Design matrix used in the least squares regression.
            - b (numpy.ndarray): Logarithmic peak ratios.
            - c (float): Intercept from least squares fit.
            - m (float): Slope from least squares fit.
    """
    zero_cross_idx = np.where(np.diff(np.sign(values.real)))[0]
    crossing_nums = np.arange(len(zero_cross_idx))+1
    time_intervals_doubled = 2*timestamps[zero_cross_idx]

    A = np.vstack([time_intervals_doubled, np.ones(len(time_intervals_doubled))]).T
    b = crossing_nums

    m, c = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = np.linalg.lstsq(A, b, rcond=None)[1][0]
    R2 = 1 - resid / (b.size * b.var())

    freq = m

    return freq, R2, A, b, c, m