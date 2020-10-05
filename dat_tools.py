# -*- coding: utf-8 -*-
"""

"""

import numpy as np

def read_node_coords(filename, nodes):
    grid_str = 'GRID'
    coords = dict()
    with open(filename,'r') as f: 
        lines = f.read().splitlines()
        i = 0
        #
        while i<len(lines) and len(coords.keys())<len(nodes):
            line_split = lines[i].split()
            if grid_str in line_split:
                if line_split[1] in nodes:
                    coords[line_split[1]] = np.array(lines[i].split()[3:6],
                                                     dtype=float)
            i += 1
            
    return coords