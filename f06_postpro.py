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
        
        
        accel_dict = dict.fromkeys(nodes, np.zeros((len(freq),3), dtype=complex))
        accel_array = np.zeros((len(nodes),len(freq),3), dtype=complex)
        freq_idx = 0
        i = 0
        # 
        while i < len(lines):
            if freq_str in lines[i][:17]:
                i += 5
                node = 0
                while lines[i][0] != '1':
                    mag = np.array(lines[i].split()[3:6],
                                   dtype=float)
                    phase = np.array(lines[i+1].split()[3:6],
                                     dtype=float) * np.pi/180
                    accel_array[node, freq_idx, :]  = mag*np.exp(1j*phase)
                    i += 2
                    node += 1
                freq_idx += 1
            i += 1
    
    node_num = 0
    for node_name in nodes:
        accel_dict[node_name] = accel_array[node_num,:,:]
        node_num += 1

    return freq, accel_dict
