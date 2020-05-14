import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def printMatrix(a):
    """Print the values of the input array as a matrix."""
    for row in a:
        for col in row:
            print("{:.2E}".format(col), end=" ")
        print("")

def plot_modes(modes):
    """Plot the columns of the input array as the
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
    """Save gif animation of the modes given by
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
        for col_B in range(modes_A.shape[1]):
            MAC_matrix[col_A, col_B] = get_MAC(modes_A[:, col_A],
                                               modes_B[:, col_B])
    return MAC_matrix

def get_max_off_diagonal(A):
    """Obtain the maximum value of an array ignoring its diagonal.
    """
    mask = np.ones(A.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    max_off_diagonal = A[mask].max()
    return max_off_diagonal

def plot_MAC(MAC, color_map, text_color):
    """Plot the input array as a MAC matrix,
    using the given colormap and text color.
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(MAC, cmap=color_map)
    fig.colorbar(cax)
    for (i, j), z in np.ndenumerate(MAC):
        ax.text(j, i, '{:.2f}'.format(z), ha='center', va='center', color=text_color)
    return ax

def get_efdd_segment(s_vec, peak_idx, mac_th):
    """Return the frequency indexes around peak_idx where the psd
    matrix having s_vec as its singular vector matrix can be
    considered as the psd of the 1DOF system, corresponding to the
    mode shape that the first singular vector takes at peak_idx.

    A MAC above the threshold mac_th is used as a criterion."""

    sdof_mode = s_vec[peak_idx, :, 0]
    lower_idx = peak_idx
    mac = 1
    while (mac>mac_th) & (lower_idx>0):
        lower_idx -= 1
        mac = get_MAC(sdof_mode, s_vec[lower_idx, :, 0])

    upper_idx = peak_idx
    mac = 1
    while (mac>mac_th) & (upper_idx<s_vec.shape[0]//2):
        upper_idx += 1
        mac = get_MAC(sdof_mode, s_vec[upper_idx, :, 0])

    return lower_idx, upper_idx