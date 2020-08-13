# -*- coding: utf-8 -*-
"""This module contains the functions necessary to perform the RFP modal
analysis method as described in Richardson et al. (1982) [1].

orthogonal_polynomials: computes the Forsythe polynomials for a given FRF

rfp: computes the modal constants for a given FRF.
"""

import numpy as np
from scipy import signal


def orthogonal_polynomials(frf, omega, weights, order):
    """Compute the Forsythe orthogonal polynomials that approximate the
    frequency response function FRF over the frequency range omega, to be used
    in the RFP method as described by Richardson and Formenti [1].
    
    Arguments:
        FRF (array): Frequency Response Function vector ()
        omega (array): Angular velocity range (rad/s)
        weights (str): "ones" for the Phi matrix, 
                        "frf" for the theta matrix.
        order (int): Polynomial degree

    Returns:
        P (array): Matrix of the polynomials evaluated at the given 
                    frequencies.
        Coeff (array): Matrix that converts Forsythe coefficients to standard
                        polynomial coefficients.

    [1] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
    response measurements using rational fraction polynomials", 1st IMAC Conference,
    Orlando, FL, 1986.
    """
    
    P = np.zeros((len(omega), order+1), dtype=complex)
    Coeff = np.zeros((order+1, order+1))
    R = np.zeros((len(omega), order+2))
    Coeff_ext = np.zeros((order+1, order+2))
    
    if weights=="ones":
        q = np.ones(len(omega)) # weighting values for phi matrix
    elif weights=="frf":
        q = np.abs(frf)**2    # weighting vuales for theta matrix
    else:
        raise Exception('Invalid weights.')

    R[:, 0] = np.zeros(len(omega))
    R[:, 1] = 1 / np.sqrt(2 * np.sum(q))

    Coeff_ext[0,1] = 1 / np.sqrt(2 * np.sum(q))

    for k in range(2, order+2):
        V_km1 = 2*np.sum(omega * R[:, k-1] * R[:, k-2] * q)
        S_k = omega * R[:, k-1] - V_km1 * R[:, k-2]
        D_k = np.sqrt(2 * np.sum(S_k**2 * q))
        
        R[:, k] = S_k / D_k
        Coeff_ext[:, k] = (1/D_k) *(-V_km1 * Coeff_ext[:, k-2] + \
                          np.concatenate([[0], Coeff_ext[:-1, k-1]]))

    j_k =  np.zeros((order+1,1), dtype=complex)
    for k in range(order+1):
        P[:, k] = 1j**k * R[:, k+1]
        j_k[k] = 1j**k

    Coeff = Coeff_ext[:, 1:]
    Coeff = (j_k @ j_k.conj().T).T * Coeff
        
    return P, Coeff

def rfp(frf, omega, n_dof):
    """Computes estimates for the modal parameters of the given FRF, in the
    frequency range given by omega, modeling it as an n_dof degrees of freedom
    system, and following the RFP method as described by Richardson and Formenti [1].

    Arguments:
        frf (numpy complex array):
            - Frequency Response Function vector (receptance).
        omega (numpy array):
            - Angular velocity range (rad/s)
        n_dof (int):
            - Number of degrees of freedom (modes) to consider for the
            estimation

    Returns:
        alpha (numpy complex array):
            - Estimated receptance FRF in the given frequency range.
        modal_params (array list):
            - Modal parameter  for the estimated FRF Modal parameter list:
                [freq_n, xi_n, modal_mag_n, modal_ang_n]

    [1] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
    response measurements using rational fraction polynomials", 1st IMAC Conference,
    Orlando, FL, 1986.
    """

    omega_norm = omega / np.max(omega) # omega normalization
    m = 2*n_dof - 1 # number of polynomial terms in numerator
    n = 2*n_dof # number of polynomial terms in denominator
    d = np.zeros(n+1) # Orthogonal denominator polynomial coefficients

    # computation of Forsythe orthogonal polynomials
    Phi, Coeff_A = orthogonal_polynomials(frf, omega_norm, 'ones', m)
    Theta, Coeff_B = orthogonal_polynomials(frf, omega_norm, 'frf', n)

    T = np.diag(frf) @ Theta[:, :-1]
    W = frf * Theta[:, -1]
    X = -2 * np.real(Phi.T.conj() @ T)
    H = 2 * np.real(Phi.T.conj() @ W)

    d[:-1] = -np.linalg.inv(np.eye(X.shape[1]) - X.T @ X) @ X.T @ H
    d[-1] = 1
    c = H - X @ d[:-1] # Orthogonal numerator polynomial coefficients

    # calculation of the estimated FRF (alpha)
    numer = Phi @ c
    denom = Theta @ d
    alpha = numer / denom

    a = np.flipud(Coeff_A @ c) # Standard polynomial numerator coefficients
    b = np.flipud(Coeff_B @ d) # Standard polynomial denominator coefficients

    # Calculation of the poles and residues
    res, pol, _ = signal.residue(a, b)
    Residuos = res[::2] * np.max(omega)
    Polos = pol[::2] * np.max(omega)

    freq_n = np.abs(Polos)/2/np.pi # Natural frequencies (rad/sec)
    xi_n = -np.real(Polos) / np.abs(Polos) # Damping ratios
    Ai = -2 * (np.real(Residuos) * np.real(Polos) + \
               np.imag(Residuos) * np.imag(Polos))
    Bi = 2 * np.real(Residuos)
    modal_const = Ai + 1j * (np.abs(Polos) * Bi)
    modal_mag_n = np.abs(modal_const) # Modal constant magnitude
    modal_ang_n = np.angle(modal_const) # Modal constant phase

    modal_params = [freq_n, xi_n, modal_mag_n, modal_ang_n] # Modal Parameters

    return modal_params, alpha

def grfp(frf, omega, n_dof):
    """Computes estimates for the modal parameters of the given FRF, in the
    frequency range given by omega, modeling it as an n_dof degrees of freedom
    system, and following the GRFP method as described by Richardson et. al
    (1986).

    Arguments:
        - frf (numpy complex array):
            - Frequency Response Function matrix. Each column contains the FRF
            for a particular DOF, with all of them corresponding the reference or
            input DOF.
        - omega (numpy array):
            - Angular velocity range (rad/s)
        - n_dof (int):
            - Number of degrees of freedom

    Returns:
        - alpha (numpy complex array):
             - Estimated receptance FRF in the given frequency range.
        - modal_params (numpy array list):
            - Modal parameters for the estimated FRF Modal parameter list:
                [freq_n, xi_n, modal_mag_n, modal_ang_n]
    """

    return None