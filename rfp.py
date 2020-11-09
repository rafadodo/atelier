# -*- coding: utf-8 -*-
"""This module contains the functions necessary to perform the RFP modal
analysis method as described in Richardson & Formenti [1] [2] [3].

forsythe_polys_rfp: computes the Forsythe polynomials for a given FRF

rfp: computes the modal constants for a given FRF.

grfp_denominator: computes the common denominator polynomial for a given
    set of FRFs.

grfp_parameters: computes the modal parameters for a set of FRFs.

[1] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
response measurements using rational fraction polynomials", 1st IMAC Conference,
Orlando, FL, 1986.

[2] Richardson, M. H. "Global frequency and damping estimates from frequency
response measurements", 4th IMAC Conference, Los Angeles, CA, 1986.

[3] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
response measurements using rational fraction polynomials", 1st IMAC Conference,
Orlando, FL, 1986.
"""

import numpy as np
from scipy import signal


def forsythe_polys_rfp(omega, weights, order):
    """Compute the Forsythe orthogonal polynomials to be used in the RFP
    method as described by Richardson and Formenti [1], to approximate a
    frequency response function FRF.
    
    Arguments:
        omega (array): Angular velocity range (rad/s)
        weights (array): Weights to use in the orthogonalization
        order (int): Maximum polynomial order

    Returns:
        P (array): Matrix of the polynomials evaluated at the given 
                    frequencies.
        Gamma (array): Matrix that converts Forsythe coefficients to standard
                        polynomial coefficients.

    [1] George E. Forsythe "Generation and Use of Orthogonal Polynomials for
    Data-Fitting With a Digital Computer", Journal of the Society for Industrial and
    Applied Mathematics, 1957.
    """

    q = weights
    P = np.zeros((len(omega), order+1), dtype=complex)
    Gamma = np.zeros((order+1, order+1))
    R = np.zeros((len(omega), order+2))
    Gamma_pre = np.zeros((order+1, order+2))

    R[:, 0] = np.zeros(len(omega))
    R[:, 1] = 1 / np.sqrt(2 * np.sum(q))

    Gamma_pre[0,1] = 1 / np.sqrt(2 * np.sum(q))

    for k in range(2, order+2):
        V_km1 = 2*np.sum(omega * R[:, k-1] * R[:, k-2] * q)
        S_k = omega * R[:, k-1] - V_km1 * R[:, k-2]
        D_k = np.sqrt(2 * np.sum(S_k**2 * q))
        
        R[:, k] = S_k / D_k
        Gamma_pre[:, k] = (1/D_k) *(-V_km1 * Gamma_pre[:, k-2] + \
                          np.concatenate([[0], Gamma_pre[:-1, k-1]]))

    j_k =  np.zeros((order+1,1), dtype=complex)
    for k in range(order+1):
        P[:, k] = 1j**k * R[:, k+1]
        j_k[k] = 1j**k

    Gamma = (j_k @ j_k.conj().T).T * Gamma_pre[:, 1:]
        
    return P, Gamma

def rfp(frf, omega, denom_order, numer_order):
    """Computes estimates for the modal parameters of the given FRF, in the
    frequency range given by omega, utilizing polynomials of order "numer_order"
    and "denom_order", and following the RFP method as described by Richardson and
    Formenti [1].

    Arguments:
        frf (numpy complex array):
            - Frequency Response Function vector (receptance).
        omega (numpy array):
            - Angular velocity range (rad/s)
        - denom_order (int):
            - Order of the denominator polynomial.
        - numer_order (int):
            - Order of the numerator polynomial.

    Returns:
        modal_params (array list):
            - Modal parameter  for the estimated FRF Modal parameter list:
                [freq_n, xi_n, modal_mag_n, modal_ang_n]
        alpha (numpy complex array):
            - Estimated receptance FRF in the given frequency range.

    [1] Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
    response measurements using rational fraction polynomials", 1st IMAC Conference,
    Orlando, FL, 1986.
    """

    omega_norm = omega / np.max(omega) # omega normalization
    m = numer_order # number of polynomial terms in numerator
    n = denom_order # number of polynomial terms in denominator
    d = np.zeros(n+1) # Orthogonal denominator polynomial coefficients

    # computation of Forsythe orthogonal polynomials
    Phi, Gamma_phi = forsythe_polys_rfp(omega_norm, np.ones(len(omega_norm)), m)
    Theta, Gamma_theta = forsythe_polys_rfp(omega_norm, np.abs(frf)**2, n)

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

    a = np.flipud(Gamma_phi @ c) # Standard polynomial numerator coefficients
    b = np.flipud(Gamma_theta @ d) # Standard polynomial denominator coefficients

    # Calculation of the poles and residues
    res, pol, _ = signal.residue(a, b)
    residues = res[::2] * np.max(omega)
    poles = pol[::2] * np.max(omega)

    freq_n = np.abs(poles)/2/np.pi # Natural frequencies (rad/sec)
    xi_n = -np.real(poles) / np.abs(poles) # Damping ratios
    modal_const = 1j*2*residues*np.imag(poles)
    modal_mag_n = np.abs(modal_const) # Modal constant magnitude
    modal_ang_n = np.angle(modal_const) # Modal constant phase

    modal_params = [freq_n, xi_n, modal_mag_n, modal_ang_n] # Modal Parameters

    return modal_params, alpha

def grfp_denominator(frf, omega, denom_order, numer_order):
    """Computes an estimate of the denominator polynomial shared by all the FRFs
    given by the columns of "frf", which correspond to the frequency range given by
    omega, utilizing polynomials of order "numer_order" and "denom_order", and
    following the GRFP method as described by Richardson [1].

    Arguments:
        - frf (numpy complex array):
            - Frequency Response Function matrix. Each column contains the FRF
            for a particular DOF, with all of them corresponding the reference or
            input DOF.
        - omega (numpy array):
            - Angular velocity range (rad/s).
        - denom_order (int):
            - Order of the denominator polynomial.
        - numer_order (int):
            - Order of the numerator polynomial.

    Returns:
        - denominator (numpy complex array):
             - Estimated denominator polynomial.
        - denominator_coeff (numpy complex array):
            - Estimated denominator polynomial coefficients.

    [1] Richardson, M. H. "Global frequency and damping estimates from frequency
    response measurements", 4th IMAC Conference, Los Angeles, CA, 1986.
    """

    n_dof = frf.shape[1] # number of frf measurements (degrees of freedom)
    w_norm = omega/np.max(omega) # normalized angular frequency range
    w_j = 1j*w_norm # complex normalized angular frequency range
    m = numer_order # number of polynomial terms in numerator
    n = denom_order # number of polynomial terms in denominator

    U = np.zeros((n_dof, n, n), dtype=complex)
    V = np.zeros((n_dof, n), dtype=complex)
    for dof in range(n_dof):
        Phi, Gamma_phi = forsythe_polys_rfp(w_norm, np.ones(len(w_norm)), m)
        Theta, Gamma_theta = forsythe_polys_rfp(w_norm, np.abs(frf[:, dof])**2, n)
        T = np.diag(frf[:, dof]) @ Theta[:, :-1]
        W = frf[:, dof] * Theta[:, -1]
        X = -2 * np.real(Phi.T.conj() @ T)
        H = 2 * np.real(Phi.T.conj() @ W)
        U[dof, :, :] = (np.eye(X.shape[1]) - X.T @ X) @ \
                        np.linalg.inv(Gamma_theta)[:-1, :-1]
        V[dof, :] = X.T @ H

    A = np.sum(U@U, axis=0)
    b = np.sum(U@V[:, :, np.newaxis], axis=0)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    denominator_coeff = np.flipud(x[:, 0])
    denominator = np.polyval(denominator_coeff, w_j)
    return denominator, denominator_coeff

def grfp_parameters(frf, omega, denom, denom_coeff, numer_order):
    """Computes an estimate of the modal parameters for each of the FRFs given by
    the columns of "frf", which correspond to the frequency range given by omega,
    following the GRFP method described by Richardson and Formenti [1]. It is assumed
    that "numer_order" is the order of the numerator polynomial, and that the common
    denominator polynomial shared by all the FRFs is given by "denom", formed by the
    coefficients "denom_coeff".

    Arguments:
        - frf (numpy complex array):
            - Frequency Response Function matrix. Each column contains the FRF
            for a particular DOF, with all of them corresponding the reference or
            input DOF.
        - omega (numpy array):
            - Angular velocity range (rad/s).
        - denom (numpy array):
            - Common denominator polynomial shared by all the FRFs.
        - denom_coeff (numpy array):
            - Coefficients that form the "denom" polynomial.
        - numer_order (int):
            - Order of the numerator polynomial.

    Returns:
        modal_params (array list):
            - Modal parameter  for the estimated FRF Modal parameter list:
                [freq_n, xi_n, modal_mag_n, modal_ang_n]

    [1] Richardson, M. H. & Formenti, D. L. "Global curve fitting of frequency response
    measurements using the Rational Fraction Polynomial method", 3rd IMAC Conference,
    Orlando, FL, 1985.
    """

    m = numer_order
    n_dof = frf.shape[1] # number of frf measurements (degrees of freedom)
    w_norm = omega/np.max(omega) # normalized angular frequency range
    w_j = 1j*w_norm # complex normalized angular frequency range
    total_poles = len(denom_coeff)-1 # number of residues and poles expected in the solution
    c = np.zeros((m+1, n_dof)) # orthogonal numerator polynomial coefficients
    # standard numerator polynomial coefficients
    numer_coef = np.zeros((m+1, n_dof), dtype=complex)
    numer = np.zeros((len(w_norm), n_dof), dtype=complex) # numerator polynomials
    alpha = np.zeros((len(w_norm), n_dof), dtype=complex) # FRF estimations
    residues_norm = np.zeros((total_poles, n_dof), dtype=complex) # frecuency-normalized residues
    poles_norm = np.zeros((total_poles, n_dof), dtype=complex) # frequency-normalized poles

    Z, Gamma_phi = forsythe_polys_rfp(w_norm, np.abs(1/denom)**2, m)
    X = np.diag(1/denom)@Z
    for dof in range(n_dof):
        c[:, dof] = 2*np.real(X.conj().T@frf[:, dof])
        numer_coef[:, dof] = np.flipud(Gamma_phi@c[:, dof])
        numer[:, dof] = np.polyval(numer_coef[:, dof], w_j)
        alpha[:, dof] = numer[:, dof] / denom
        residues_norm[:, dof], poles_norm[:, dof], _ = signal.residue(numer_coef[:, dof],
                                                                      denom_coeff)

    xi_n_raw = -np.real(poles_norm*np.max(omega))/np.abs(poles_norm*np.max(omega))
    # modes with unitary damping coefficient are discarded
    physical_modes_idx = (abs(xi_n_raw[:, 0])!=1)
    residues = residues_norm[physical_modes_idx, :][::2]*np.max(omega)
    poles = poles_norm[physical_modes_idx, :][::2]*np.max(omega)

    freq_n = np.abs(poles[:, 0])/2/np.pi
    xi_n = -np.real(poles[:, 0])/np.abs(poles[:, 0])
    modal_const = 1j*2*residues*np.imag(poles)
    modal_mag_n = np.abs(modal_const) # Modal constant magnitude
    modal_ang_n = np.angle(modal_const) # Modal constant phase

    modal_params = [freq_n, xi_n, modal_mag_n, modal_ang_n]
    return modal_params, alpha