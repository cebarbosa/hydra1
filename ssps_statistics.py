# -*- coding: utf-8 -*-
"""

Created on 27/10/15

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import matplotlib.pyplot as plt

from config import *
from plot_lick_radius import mask_slits

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    mask_slits()
    data = np.loadtxt("results_masked.tab", usecols=(3,4,69,72,75,84))
    specs, sn = np.loadtxt("results_masked.tab", usecols=(0,14), dtype=str).T
    ads = np.loadtxt("results_masked.tab", usecols=(87,88,89,90)).T
    sn = sn.astype(float)
    #########################################################################
    # Filtering data
    # Removing inner halo
    specs = specs[data[:,0] > 1.5 * re]
    sn = sn[data[:,0] > 1.5 * re]
    ads = ads.T[data[:,0] > 1.5 * re].T
    data = data[data[:,0] > 1.5 * re]
    plt.pause(0.001)
    plt.show(block=1)
    idx = np.arange(len(data))
    ne = np.logical_and(data[:,1] > 0, data[:,1] < 90)
    idx_ne = idx[ne]
    idx_others = idx[~ne]
    data= data[:,2:].T
    parameters = [r" log Age", "[Z/H]", "[alpha/Fe]", "Fe/H"]
    for j, (p,a) in enumerate(zip(data, ads)):
        print "{0:20s}{1:5s}{2:8s}{3:8s}".format(parameters[j], "N", "MEAN", "SIGMA")
        sample = ["All halo", "NE-quadrant", "Other quadrants"]
        for k, i in enumerate([idx, idx_ne, idx_others]):
            ii = np.logical_and(np.isfinite(p[i]), a[i] < 15)
            if k==1 and j==1:
                print specs[i][ii]
            print "{0:15s}".format(sample[k]), "{0:5}".format(len(p[i][ii])),
            mad = 1.4826 * np.median(np.abs(p[i][ii] - np.median(p[i][ii])))
            print "{0:8.2f}".format(np.median(p[i][ii])), "{0:8.2f}".format(mad)
        print "\n"
    #     ax = plt.subplot(4,1,j+1)
    #     # ax.hist(p)
    #     ax.hist(p[idx_others])
    #     ax.hist(p[idx_ne])
    # plt.pause(0.001)
    # plt.show(block=1)
    raw_input()
    # for i in [idx, idx_ne, idx_others]

