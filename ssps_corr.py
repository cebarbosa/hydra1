# -*- coding: utf-8 -*-
"""

Created on 09/12/15

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import matplotlib.pyplot as plt

from config import *
from maps import match_data

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    filenames = ["age_dist_sym.txt", "metal_dist_sym.txt",
                 "alpha_dist_sym.txt"]
    fig = plt.figure(1)
    for (i,j) in [(0,1), (1,2), (2,0)]:
        s1 = np.loadtxt(filenames[i], dtype=str)
        s2 = np.loadtxt(filenames[j], dtype=str)
        sref = np.intersect1d(s1[:,0], s2[:,0]).tolist()
        s1 = match_data(s1[:,0].tolist(), sref, s1)
        s2 = match_data(s2[:,0].tolist(), sref, s2)
        d1 = s1[:,1].astype(float)
        d2 = s2[:,1].astype(float)
        ax = plt.subplot(1,3,i+1)
        ax.plot(d1, d2, "ok")
    plt.pause(0.001)
    plt.show(block=1)




