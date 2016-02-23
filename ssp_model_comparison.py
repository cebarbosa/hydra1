# -*- coding: utf-8 -*-
"""

Created on 26/11/15

@author: Carlos Eduardo Barbosa

Plot the comparison of the SSP for MILES II and Thomas et al. 2010 models
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from config import *
from mcmc_model import get_model_lims



if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    data1 = np.loadtxt("populations.txt", usecols=(1,2,3,5,6,7,9,10,11))
    data2 = np.loadtxt("populations_miles.txt", usecols=(1,2,3,5,6,7,9,10,11))
    lims, ranges = get_model_lims(os.path.join(tables_dir, "MILESII.txt"))
    ranges[0] = [9.8,10.2]
    ranges[1] = [-1.5, 0.6]
    err_cut = np.array([0.2, 0.7, 0.22])
    fig = plt.figure(1, figsize=(14, 4.5))
    labels = [r"$\log$ Age (yr)", r"[Z/H]", r"[$\alpha$/Fe]"]
    plt.subplots_adjust(left=0.065, right=0.98, bottom=0.13, top=0.97,
                        wspace=0.25)
    for i,j in enumerate([0,3,6]):
        idx1 = np.intersect1d(np.where(data1[:,j] >= ranges[i,0])[0],
                             np.where(data1[:,j] <= ranges[i,1])[0])
        merr1 = 0.5 * np.abs((data1[:,j+1] - data1[:,j+2]))
        merr2 = 0.5 * np.abs((data2[:,j+1] - data2[:,j+2]))
        merr = np.maximum(merr1, merr2)
        idx2 = np.where(merr <= err_cut[i])[0]
        idx = np.intersect1d(idx1, idx2)
        ax = plt.subplot(1,3,i+1)
        ax.minorticks_on()
        x = data1[idx][:,j]
        y =  data2[idx][:,j]
        xerr = [x - data1[idx][:,j+1], data1[idx][:,j+2] - x]
        yerr = [y - data2[idx][:,j+1], data2[idx][:,j+2] - y]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, ecolor="0.6", capsize=0,
                    fmt="o", ms=8)
        ax.set_xlim(ranges[i,0], ranges[i,1])
        ax.set_ylim(ranges[i,0], ranges[i,1])
        xx = np.linspace(ranges[i,0], ranges[i,1], 100)
        ax.plot(xx, xx, "--k")
        ax.plot(xx, xx + np.sqrt(np.mean((x-y)**2)), ":k")
        ax.plot(xx, xx - np.std(x - y), ":k")
        ax.set_xlabel("{0} -- Thomas et al. 2011".format(labels[i]))
        ax.set_ylabel("{0} -- Vazdekis et al. 2015".format(labels[i]))
        print np.sqrt(np.mean((x-y)**2))
    plt.savefig(os.path.join(figures_dir, "models_comparison.png"))


