# -*- coding: utf-8 -*-
"""
This module contains functions that extract data from NASTRAN '.f06' output files
"""
import numpy as np

def read_nodes_accel(filename):
    """ """
    freq_str = '      FREQUENCY ='
    freq = np.array([])
    accel = dict()
    nodes = list()
    with open(filename,'r') as f: 
        lines = f.read().splitlines()
        first_freq = True
        i = 0
        #
        while i < len(lines):
            if freq_str in lines[i][:17]:
                freq =  np.append(freq, float(lines[i].split()[2]))
                i += 5
                if first_freq:
                    first_freq = False
                    while lines[i][0] != '1':
                        nodes.append(lines[i].split()[1])
                        i += 2
            i += 1

        for node in nodes:
            accel[node] = np.zeros((len(freq),3), dtype=complex)
        freq_idx = 0
        i = 0
        # 
        while i < len(lines):
            if freq_str in lines[i][:17]:
                i += 5
                while lines[i][0] != '1':
                    node = lines[i].split()[1]
                    mag = np.array(lines[i].split()[3:6],
                                   dtype=float)
                    phase = np.array(lines[i+1].split()[3:6],
                                     dtype=float) * np.pi/180
                    accel[node][freq_idx, :]  = mag*np.exp(1j*phase)
                    i += 2
                freq_idx += 1
            i += 1

    return freq, accel, nodes

def read_modeshapes(filename, nodes):
    """"Read mode shapes from an f06 file corresponding to a SOL 103 NASTRAN
    simulation, assuming real displacement is printed."""
    freq_str = '          CYCLES ='
    N_DOF = 3 # Degrees of freedom to consider for modeshapes: X Y Z
    freqs = np.array([])
    with open(filename,'r') as f:
        lines = f.read().splitlines()
        i = 0
        # Read all modal frequencies
        while i < len(lines):
            if lines[i][:18] == freq_str:
                # Check it is not a repeated frequency
                if float(lines[i].split()[2]) not in freqs:
                    freqs =  np.append(freqs, float(lines[i].split()[2]))
            i += 1

        modeshapes = np.zeros((len(freqs), N_DOF*len(nodes)))
        freq_idx = 0
        i = 0
        # Read mode shape for each frequency
        while i < len(lines):
            if lines[i][:18] == freq_str and freq_idx<len(freqs):
                # Check it is the currently expected frequency
                if float(lines[i].split()[2]) == freqs[freq_idx]:
                    i += 3
                    dof = 0
                    while lines[i][0] != '1':
                        node = lines[i].split()[0]
                        if node in nodes:
                            node_disp = np.array(lines[i].split()[2:5], dtype=float)
                            modeshapes[freq_idx, dof:dof+N_DOF]  = node_disp
                            dof += N_DOF
                        i += 1
                    freq_idx += 1
            i += 1

    return freqs, modeshapes