# -*- coding: utf-8 -*-
"""

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import matplotlib.pyplot as plt

from config import *

if __name__ == "__main__":
    wdir = os.path.join(home, "single2")
    os.chdir(wdir)
    sn = np.loadtxt("results.tab", usecols=(14,))
    data = np.loadtxt("results.tab", usecols=(69,72,75,84)).T
    errp = np.loadtxt("results.tab", usecols=(70,73,76,85)).T
    errm = np.loadtxt("results.tab", usecols=(71,74,77,86)).T
    err = np.abs(errp - errm) / 2
    fig = plt.figure(1)
    for i in range(4):
        ax = plt.subplot(4,1,i+1)
        ax.plot(sn, err[i],'ok' )
    plt.pause(0.001)
    plt.show()

