# -*- coding: utf-8 -*-
"""


"""
import numpy as np

def read_nodes_accel(filename):
    freq_str = '      FREQUENCY ='
    freq = np.array([])
    nodes = list()
    with open(filename,'r') as f: 
        lines = f.read().splitlines()
        first_freq = True
        i = 0
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
        
        
        accel_mag = dict.fromkeys(nodes, np.zeros((len(freq),3)))
        accel_ph = dict.fromkeys(nodes, np.zeros((len(freq),3)))
        freq_idx = 0
        i = 0
        while i < len(lines):
            if freq_str in lines[i][:17]:
                i += 5
                while lines[i][0] != '1':
                    node = lines[i].split()[1]
                    accel_mag[node][freq_idx,:]  = np.array(lines[i].split()[3:6])
                    accel_ph[node][freq_idx,:]  = np.array(lines[i+1].split()[3:6])
                    i += 1
                freq_idx += 1
            i += 1

    return freq, accel_mag, accel_ph
