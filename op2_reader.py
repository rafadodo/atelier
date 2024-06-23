# -*- coding: utf-8 -*-
"""This module contains the functions necessary to...
"""

import pyNastran
from pyNastran.op2.op2 import OP2
import matplotlib.pyplot as plt
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename

Tk().withdraw()
accepted_filetypes = [('OP2 files', '*.OP2')]
OP2_FILENAME = askopenfilename(
    title='Select an OP2 file...',
    filetypes=accepted_filetypes,
)
# OP2_FILENAME = r'C:\Users\dodor\OneDrive\Documentos\Simu tests\sate\sate_subassembly_sinex_200hz_plot.op2'
DIRS = {'X':0, 'Y':1, 'Z':2}

model = OP2()
model.read_op2(OP2_FILENAME)
#print(model.get_op2_stats())



subcase_label = 1
dir = 'X'
node = 11
accel = model.accelerations
nodes = accel[subcase_label].node_gridtype[:,0]
node_idx = np.where(nodes==node)[0]
freqs = accel[1].freqs
node_TF = abs((accel[1].data[:, node_idx, DIRS[dir]]))

plt.figure()
plt.semilogy(freqs, node_TF)
plt.title('Node Acceleration Transfer Function')
plt.ylim([np.min(node_TF), np.max(node_TF)])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Acceleration Transfer Function [g/g]')
plt.grid()
plt.show()