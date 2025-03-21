# -*- coding: utf-8 -*-
"""
This module contains functions that assist in dealing with MAC (Modal Assurance Criterion) values
and matrices.

Functions:
- get_mac_value(mode_A, mode_B): Computes the MAC value between two mode shape vectors.
- get_mac_matrix(modes_A, modes_B): Computes the MAC matrix between two sets of mode shapes.
- plot_mac(mac_matrix, color_map, text_color, title_str, labels): Plots a given MAC matrix.
"""
import numpy as np
import matplotlib.pyplot as plt


def get_mac_value(mode_A, mode_B):
    """Obtain the MAC for the mode shapes defined by column vectors A and B.

    Parameters:
        mode_A (numpy.ndarray): A column vector representing a mode shape.
        mode_B (numpy.ndarray): A column vector representing another mode shape.

    Returns:
        float: The MAC value
    """
    mac = abs(mode_A.T @ mode_B.conj())**2 / \
          (abs(mode_A.T @ mode_A.conj()) * abs(mode_B.T @ mode_B.conj()))

    return mac


def get_mac_matrix(modes_A, modes_B):
    """Compute the MAC matrix for two sets of mode shapes.

    Parameters:
        modes_A (numpy.ndarray): A matrix where each column represents a mode shape.
        modes_B (numpy.ndarray): Another matrix where each column represents a mode shape.

    Returns:
        numpy.ndarray: A MAC matrix where each element (i, j) represents the MAC value between
                       mode i from modes_A and mode j from modes_B.
    """
    mac_matrix = np.zeros((modes_A.shape[1], modes_B.shape[1]))
    for col_A in range(modes_A.shape[1]):
        for col_B in range(modes_B.shape[1]):
            mac_matrix[col_A, col_B] = get_mac_value(modes_A[:, col_A],
                                               modes_B[:, col_B])

    return mac_matrix


def plot_mac(mac_matrix, color_map, text_color, title_str='MAC', labels=['Mode', 'Mode']):
    """Plot a given Modal Assurance Criterion (MAC) matrix.

    Parameters:
        MAC (numpy.ndarray): The MAC matrix to plot.
        color_map (str): The colormap to use for the MAC values display.
        text_color (str): Color for the text displaying MAC values on the plot.
        title_str (str, optional): Title for the MAC plot.
        labels (list of str, optional): Labels for the rows and columns of the matrix.

    Returns:
        matplotlib.axes.Axes: The axes object of the resulting plot.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(mac_matrix, cmap=color_map)
    plt.title(title_str, y=1.15)
    plt.ylabel(labels[0])
    plt.xlabel(labels[1])
    ax.xaxis.set_label_position('top')

    ticks = np.arange(mac_matrix.shape[0])
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(ticks + 1)
    ax.set_yticklabels(ticks + 1)

    for (i, j), z in np.ndenumerate(mac_matrix):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center', color=text_color)

    plt.tight_layout()

    return ax