import numpy as np
import matplotlib.pyplot as plt

def get_MAC(mode_A, mode_B):
    """Obtain the MAC for the modes defined by
    column vectors A and B.
    """
    MAC = abs(mode_A.T @ mode_B.conj())**2 / \
          (abs(mode_A.T @ mode_A.conj()) * abs(mode_B.T @ mode_B.conj()))
    return MAC
    
def get_MAC_matrix(modes_A, modes_B):
    """Obtain the MAC matrix for the modes represented by
    the columns of matrices modes_A and modes_B.
    Rows of the MAC matrix correspond to modes from modes_A.
    """
    MAC_matrix = np.zeros((modes_A.shape[1], modes_B.shape[1]))
    for col_A in range(modes_A.shape[1]):
        for col_B in range(modes_B.shape[1]):
            MAC_matrix[col_A, col_B] = get_MAC(modes_A[:, col_A],
                                               modes_B[:, col_B])
    return MAC_matrix

def plot_MAC(MAC, color_map, text_color, title_str='MAC', labels=['', '']):
    """Plot a given Modal Assurance Criterion matrix.

    Arguments:
        MAC (array): MAC matrix to plot.
        labels (string list): Labels for the rows and columns of the matrix respectively.
        color_map (): Color map to use for MAC values display.
        text_color (): Text color to use for MAC values display.
        title_str (): Title for the MAC plot.

    Returns:
        ax (matplotlib figure axes): axes for the resulting plot.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(MAC, cmap=color_map)
    plt.title(title_str, y=1.15)
    plt.ylabel(labels[0])
    plt.xlabel(labels[1])
    ax.xaxis.set_label_position('top')
    ticks_lst = ['']+list(np.arange(MAC.shape[0])+1)
    ax.set_xticklabels(ticks_lst)
    ax.set_yticklabels(ticks_lst)
    plt.tight_layout()
    for (i, j), z in np.ndenumerate(MAC):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center', color=text_color)
    return ax