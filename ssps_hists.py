# -*- coding: utf-8 -*-
"""

Created on 27/10/15

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import matplotlib.pyplot as plt

from config import *

def make_fe_traces(specs):
    """ Create artificial [Fe/H] traces. """
    for spec in specs:
        print spec
        path = os.path.join(os.getcwd(), spec.replace(".fits", "_db"),
                                "Chain_0")
        file1 = os.path.join(path, "alpha_dist.txt")
        file2 = os.path.join(path, "metal_dist.txt")
        fileout = os.path.join(path, "iron_dist.txt")
        d1 = np.loadtxt(file1, usecols=(0,))
        d2 = np.loadtxt(file2, usecols=(0,))
        feh = d2 - 0.94 * d1
        np.savetxt(fileout, feh)

def get_data(specs, filename):
    """Stack results from traces of a given spectra"""
    for i,spec in enumerate(specs):
        path = os.path.join(os.getcwd(), spec.replace(".fits", "_db"),
                                "Chain_0")
        filename = os.path.join(path, filename)
        d = np.loadtxt(filename, usecols=(0,))
        if i == 0:
            data = np.zeros((len(specs), len(d)))
        data[i] = d
    return data

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    # Cols: 0:R, 1:PA, 2-5: SSPs, 6:S/N, 7-10:AD test
    data = np.loadtxt("results_masked.tab", usecols=(3,4,69,72,75,84,14,87,
                                                     88,89,90))
    specs = np.loadtxt("results_masked.tab", usecols=(0,), dtype=str)
    # make_fe_traces(specs)
    ##########################################################################
    # Filtering data for inner radius
    specs = specs[data[:,0] > re]
    data = data[data[:,0] > re]
    ##########################################################################
    # Index for three samples
    idx = np.arange(len(data))
    ne = np.logical_and(data[:,1] > 0, data[:,1] < 90)
    idx_ne = idx[ne]
    idx_others = idx[~ne]
    ##########################################################################
    ssps = data[:,2:6].T
    sn = data[:,6]
    ads = data[:,7:].T
    parameters = [r" log Age", "[Z/H]", "[alpha/Fe]", "Fe/H"]
    filenames = ["age_dist.txt", "metal_dist.txt", "alpha_dist.txt",
                 "iron_dist.txt"]
    for j, (values, test) in enumerate(zip(ssps,ads)):
        # data = get_data(specs, filenames[j])
        for k, i in enumerate([idx, idx_others, idx_ne]):
            t = test[i]
            v = values[i]
            cond = np.where(t < 15)[0]
            if k == 0:
                bins= np.histogram(v[cond], bins=15)[1]
            else:
                ax = plt.subplot(8,1,2*j+k)
                weights = np.ones_like(v[cond])/float(len(v[cond]))
                ax.hist(v[cond], bins=bins, normed=1, weights=weights)
                ax.set_xlim(bins[0], bins[-1])
    plt.pause(0.001)
    plt.show(block=1)