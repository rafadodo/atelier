# -*- coding: utf-8 -*-
"""This module contains the functions necessary to perform the RFP modal
analysis method as described in Richardson et al. (1982) [1].

orthogonal_polynomials: computes the Forsythe polynomials for a given FRF

rfp: computes the modal constants for a given FRF.
"""

import numpy as np


def orthogonal_polynomials(frf, omega, weights, order):
    """Compute the Forsythe orthogonal polynomials that approximate the
    frequency response function FRF over the frequency range omega, to be used
    in the RFP method as described by Richardson et. al (1982).
    
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
    """
    
    P = np.zeros((len(omega), order))
    Coeff = np.zeros((order+1, order+2))
    
    if weights=="ones":
    	q = np.ones(len(omega)) # weighting values for phi matrix
    elif weights=="frf":
    	q = (np.abs(frf))**2    # weighting vuales for theta matrix
    else:
    	raise Exception('invalid weights.')

    Coeff[0,1] = 1 / np.sqrt(2 * np.sum(q))
    
    R = np.zeros(len(omega), order+2)
    R[:, 0] = np.zeros(len(omega))
    R[:, 1] = 1 / np.sqrt(2 * np.sum(q))

    for k in range(2, order+1):
        V_km1 = 2*np.sum(omega * R[:, k-1] * R[:, k-2] * q)
        S_k = omega * R[:, k+1] - V_km1 * R[:, k-2]
        D_k = np.sqrt(2 * np.sum(S_k**2 * q))
        
        R[:, k] = S_k / D_k
        Coeff[:, k] = -V_km1 * Coeff[:, k-2] 
        Coeff[1:k-1, k] += Coeff[1:k-1, k-1] # copyyy
        Coeff[:, k] = Coeff[:, k] / D_k
        
    
    j_k =  np.zeros(order+1)
    for k in range(order+1):
        P[:, k] = 1j**k * R[:, k+1]
        j_k[k] = 1j**k
        
    Coeff = (j_k @ j_k.T) @ Coeff
        
    return P, Coeff

def rfp(frf, omega, n_modes):
    """"""
    return None