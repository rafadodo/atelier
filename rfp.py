# -*- coding: utf-8 -*-
"""
This module contains functions for modal parameter estimation using the Rational
Fraction Polynomials (RFP) method, as described in the corresponding literature
(see references inside each function).

Functions:
- forsythe_polys_rfp(omega, weights, order): Computes Forsythe polynomials for an FRF.
- rfp(frf, omega, denom_order, numer_order): Computes modal constants for an FRF.
- grfp_denominator(frf, omega, denom_order, numer_order): Computes the common denominator
  polynomial for a given set of FRFs.
- grfp_parameters(frf, omega, denom, denom_coeff, numer_order): Computes modal parameters
  from a set of FRFs.
"""

import numpy as np
from scipy import signal


def forsythe_polys_rfp(omega, weights, order):
    """Computes Forsythe orthogonal polynomials following the formulation proposed by Richardson
    and Formenti for the Rational Fraction Polynomials method (see References below).

    Parameters:
        omega (numpy.ndarray, shape (n_freq,)): Angular frequency range (rad/s).
        weights (numpy.ndarray, shape (n_freq,)): Weights used for orthogonalization.
        order (int): Maximum polynomial order.

    Returns:
        tuple:
            - P (numpy.ndarray, shape (n_freq, order+1)): Matrix of polynomials evaluated
              at the given frequencies.
            - Gamma (numpy.ndarray, shape (order+1, order+1)): Transformation matrix
              converting Forsythe coefficients to standard polynomial coefficients.

    References:
        - George E. Forsythe "Generation and Use of Orthogonal Polynomials for
            Data-Fitting With a Digital Computer", Journal of the Society for Industrial and
            Applied Mathematics, 1957.
        - Richardson, M. H. & Formenti D. L. "Parameter estimation from frequency
            response measurements using rational fraction polynomials", 1st IMAC Conference,
            Orlando, FL, 1982.
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
    """Estimates modal parameters from a given Frequency Response Function (FRF)
    using the Rational Fraction Polynomials (RFP) method (see Reference below).

    Parameters:
        frf (numpy.ndarray, shape (n_freq,)): Frequency Response Function (receptance).
        omega (numpy.ndarray, shape (n_freq,)): Angular frequency range (rad/s).
        denom_order (int): Order of the denominator polynomial.
        numer_order (int): Order of the numerator polynomial.

    Returns:
        tuple:
            - modal_params (list of numpy.ndarray): List containing estimated modal parameters:
                - freq_n (numpy.ndarray): Natural frequencies (Hz).
                - xi_n (numpy.ndarray): Damping ratios.
                - modal_mag_n (numpy.ndarray): Modal constant magnitudes.
                - modal_ang_n (numpy.ndarray): Modal constant phases.
            - alpha (numpy.ndarray, shape (n_freq,)): Estimated FRF in the given frequency range.

    Reference:
        M. H. Richardson & D. L. Formenti, "Parameter estimation from frequency
        response measurements using rational fraction polynomials", 1st IMAC Conference,
        Orlando, FL, 1982.
    """
    omega_norm = omega / np.max(omega)
    m = numer_order
    n = denom_order
    d = np.zeros(n+1) # Orthogonal denominator polynomial coefficients

    Phi, Gamma_phi = forsythe_polys_rfp(omega_norm, np.ones(len(omega_norm)), m)
    Theta, Gamma_theta = forsythe_polys_rfp(omega_norm, np.abs(frf)**2, n)

    T = np.diag(frf) @ Theta[:, :-1]
    W = frf * Theta[:, -1]
    X = -2 * np.real(Phi.T.conj() @ T)
    H = 2 * np.real(Phi.T.conj() @ W)

    d[:-1] = -np.linalg.inv(np.eye(X.shape[1]) - X.T @ X) @ X.T @ H
    d[-1] = 1
    c = H - X @ d[:-1] # Orthogonal numerator polynomial coefficients

    # Calculation of the estimated FRF (alpha)
    numer = Phi @ c
    denom = Theta @ d
    alpha = numer / denom

    a = np.flipud(Gamma_phi @ c) # Standard polynomial numerator coefficients
    b = np.flipud(Gamma_theta @ d) # Standard polynomial denominator coefficients

    # Calculation of the poles and residues
    res, pol, _ = signal.residue(a, b)
    residues = res[::2] * np.max(omega)
    poles = pol[::2] * np.max(omega)

    freq_n = np.abs(poles)/2/np.pi
    xi_n = -np.real(poles) / np.abs(poles)
    modal_const = 1j*2*residues*np.imag(poles)
    modal_mag_n = np.abs(modal_const)
    modal_ang_n = np.angle(modal_const)

    modal_params = [freq_n, xi_n, modal_mag_n, modal_ang_n]

    return modal_params, alpha

def grfp_denominator(frf, omega, denom_order, numer_order):
    """Estimates the common denominator polynomial for a set of FRFs using the
    Global Rational Fraction Polynomial (GRFP) method (see Reference below).

    Parameters:
        frf (numpy.ndarray, shape (n_freq, n_dof)): FRF matrix where each column represents
            the FRF for a specific DOF, all corresponding to the reference/input DOF.
        omega (numpy.ndarray, shape (n_freq,)): Angular frequency range (rad/s).
        denom_order (int): Order of the denominator polynomial.
        numer_order (int): Order of the numerator polynomial.

    Returns:
        tuple:
            - denominator (numpy.ndarray, shape (n_freq,)): Estimated common denominator polynomial.
            - denominator_coeff (numpy.ndarray, shape (denom_order+1,)): Estimated denominator
              polynomial coefficients.

    Reference:
        M. H. Richardson, "Global frequency and damping estimates from frequency
        response measurements", 4th IMAC Conference, Los Angeles, CA, 1986.
    """
    n_dof = frf.shape[1]
    w_norm = omega/np.max(omega)
    w_j = 1j*w_norm
    m = numer_order
    n = denom_order

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
    """Computes modal parameters for a set of FRFs using the GRFP method (see Reference below).

    Parameters:
        frf (numpy.ndarray, shape (n_freq, n_dof)): FRF matrix where each column represents
            the FRF for a specific DOF, all corresponding to the reference/input DOF.
        omega (numpy.ndarray, shape (n_freq,)): Angular frequency range (rad/s).
        denom (numpy.ndarray, shape (n_freq,)): Common denominator polynomial shared by all FRFs.
        denom_coeff (numpy.ndarray, shape (denom_order+1,)): Coefficients of the denominator polynomial.
        numer_order (int): Order of the numerator polynomial.

    Returns:
        tuple:
            - modal_params (list of numpy.ndarray): List containing estimated modal parameters:
                - freq_n (numpy.ndarray): Natural frequencies (Hz).
                - xi_n (numpy.ndarray): Damping ratios.
                - modal_mag_n (numpy.ndarray): Modal constant magnitudes.
                - modal_ang_n (numpy.ndarray): Modal constant phases.
            - alpha (numpy.ndarray, shape (n_freq, n_dof)): Estimated FRF matrix.

    Reference:
        M. H. Richardson & D. L. Formenti, "Global curve fitting of frequency response
        measurements using the Rational Fraction Polynomial method", 3rd IMAC Conference,
        Orlando, FL, 1985.
    """
    m = numer_order
    n_dof = frf.shape[1]
    w_norm = omega/np.max(omega)
    w_j = 1j*w_norm
    total_poles = len(denom_coeff)-1
    c = np.zeros((m+1, n_dof)) # orthogonal numerator polynomial coefficients
    # standard numerator polynomial coefficients
    numer_coef = np.zeros((m+1, n_dof), dtype=complex)
    numer = np.zeros((len(w_norm), n_dof), dtype=complex)
    alpha = np.zeros((len(w_norm), n_dof), dtype=complex)
    residues_norm = np.zeros((total_poles, n_dof), dtype=complex)
    poles_norm = np.zeros((total_poles, n_dof), dtype=complex)

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
    modal_mag_n = np.abs(modal_const)
    modal_ang_n = np.angle(modal_const)

    modal_params = [freq_n, xi_n, modal_mag_n, modal_ang_n]
    return modal_params, alpha