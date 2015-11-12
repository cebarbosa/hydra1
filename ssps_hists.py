# -*- coding: utf-8 -*-
"""

Created on 27/10/15

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import matplotlib.pyplot as plt
# from sklearn.mixture import GMM
from matplotlib.mlab import normpdf

from config import *
from mcmc_analysis import gmm

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
    sn = np.loadtxt("results_masked.tab", usecols=(14,))
    # Calculating maximum errors
    data1 = np.loadtxt("results_masked.tab", usecols=(69,72,75,84)).T
    errp = np.loadtxt("results_masked.tab", usecols=(70,73,76,85)).T
    errm = np.loadtxt("results_masked.tab", usecols=(71,74,77,86)).T
    err = np.maximum(np.abs(errp - data1), np.abs(errm - data1)).T
    err_cut = np.array([0.2, 0.4, 0.2, 0.6])
    ##########################################################################
    # Filtering data for inner radius
    specs = specs[data[:,0] > re]
    err = err[data[:,0] > re].T
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
    parameters = [r" log Age", "[Z/H]", r"[$\alpha$/Fe]", "[Fe/H]"]
    filenames = ["age_dist.txt", "metal_dist.txt", "alpha_dist.txt",
                 "iron_dist.txt"]
    colors = ["w", "r", "gray"]
    labels = ["All", "Off-Centered \nHalo (N=", "Symmetric Halo\n(N="]
    fig = plt.figure(1, figsize=(6,7))
    plt.subplots_adjust(bottom=0.08, right=0.98, top=0.98, left=0.05)
    xlims = [[10,10.2], [-2,1], [-.3,0.5], [-1.5,1]]
    for j, (values, error) in enumerate(zip(ssps,err)):
        print parameters[j]
        for k, i in enumerate([idx, idx_ne, idx_others]):
            print labels[k]
            e = error[i]
            v = values[i]
            cond = np.where(e < err_cut[j])[0]
            if k == 0:
                bins= np.histogram(v[cond], bins=10, range=xlims[j])[1]
            else:
                ax = plt.subplot(2,2,j+1)
                plt.locator_params(nbins=5)
                ax.minorticks_on()
                weights = np.ones_like(v[cond])/float(len(v[cond]))
                ax.hist(v[cond], bins=bins, normed=True, weights=None,
                        alpha=0.5, histtype='stepfilled', color=colors[k],
                        edgecolor="none", label=labels[k]+str(len(v[cond]))+")")
                ax.set_xlim(*xlims[j])
                ax.set_xlabel(parameters[j])
                # ax.axvline(v[cond].mean(), c=colors[k], lw=2, ls="--")
                plt.legend(loc='upper left', prop={'size':10}, frameon=False)
                # d = gmm(v[cond])
                #
                # d.best = d.models[np.argmin(d.BIC)]
                # x = np.linspace(xlims[j][0], xlims[j][1], 100)
                # for m,w,c in zip(d.best.means_, d.best.weights_,
                #                  d.best.covars_):
                #     y = w * normpdf(x, m, np.sqrt(c))[0]
                #     ax.plot(x, y, "--", c=colors[k])

    plt.pause(0.001)
    plt.savefig(os.path.join(os.getcwd(), "figs/hist_outer.png"))
    # plt.show(block=1)