# -*- coding: utf-8 -*-
"""
This module contains functions that extract data from NASTRAN '.dat' output files
"""
import numpy as np

def nastran_float(s):
    '''Convert a string containing a number in NASTRAN format, that is, with
    only an '+' or '-' indicating exponential notation, and no 'e', into python
    format, by adding said 'e'.'''
    s = s.replace('-','e-')
    s = s.replace('+','e+')
    if s[0] == 'e':
        s = s[1:]
    return s

def read_node_coords(filename, nodes):
    grid_str = 'GRID'
    chunk_size = 8
    coords = dict()
    with open(filename,'r') as f: 
        lines = f.read().splitlines()
        i = 0
        #
        while i<len(lines) and len(coords.keys())<len(nodes):
            line = lines[i]
            line_split = line.split()
            if grid_str in line_split:
                if line_split[1] in nodes:
                    node = line_split[1]
                    chunks = list()
                    for c in range(0, len(line), chunk_size):
                        chunks.append(
                                nastran_float(line[c:c+chunk_size].strip())
                                )
                    coords[node] = np.array(chunks[3:6], dtype=float)
            i += 1
            
    return coords