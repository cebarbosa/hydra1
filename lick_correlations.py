# -*- coding: utf-8 -*-
"""

Created on 01/04/16

@author: Carlos Eduardo Barbosa

Make index-index correlations

"""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr

from config import *

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    cols = np.array([39,41,43, 45, 47,49,51,53,55])
    table = "results.tab"
    data = np.loadtxt(table, usecols=cols).T
    error = np.loadtxt(table, usecols=cols+1).T
    cols2 = np.array([15,16,17,18,19,20,21,22,23])
    table2 = os.path.join(tables_dir, "tmj.dat")
    data2 = np.loadtxt(table2, usecols=cols2).T
    labels = ["H beta (\AA)", "Fe5015 (\AA)", "Mg 1 (mag)", "Mg 2 (mag)",
              "Mg b (\AA)", "Fe5270 (\AA)", "Fe5335 (\AA)", "Fe5406 (\AA)",
              "Fe5709 (\AA)"]
    medians = np.repeat(np.nanmedian(data, axis=1),
                        len(data[0])).reshape(data.shape)
    mad = 1.48 * np.nanmedian(np.abs(data - medians), axis=1)
    center = medians[:,0]
    fig = plt.figure(1, figsize=(16,16))
    c = np.arange(len(cols))
    c = np.array(np.meshgrid(c, c)).T.reshape(-1,2)
    gs = gridspec.GridSpec(len(cols),len(cols))
    gs.update(left=0.055,right=0.99, top=0.99, bottom=0.04, wspace=0.05,
              hspace=0.05)
    for l,(j,i) in enumerate(c):
        ax = plt.subplot(gs[l])
        ax.minorticks_on()
        idx = np.logical_and(np.isfinite(data[i]), np.isfinite(data[j]))
        idxclip1 = np.logical_and(data[i] < center[i] + 5 * mad[i],
                                 data[i] > center[i] - 5 * mad[i])
        idxclip2 = np.logical_and(data[j] < center[j] + 5 * mad[j],
                                 data[j] > center[j] - 5 * mad[j])
        idx1 = np.logical_and(idxclip1, idxclip2)
        idx = np.logical_and(idx1, idx)
        r, p = spearmanr(data[i][idx], data[j][idx])
        fmt = "xk" if p > 0.03 else "or"
        mec = "k" if p > 0.03 else "0.7"
        ax.plot(data2[i], data2[j], "s", mec="none",color="0.5")
        ax.errorbar(data[i], data[j], xerr=error[i], yerr=error[j],
                    fmt="none", ecolor="0.7", capsize=0)
        ax.plot(data[i], data[j], fmt, mec=mec,
                label="r={0:.2f}, p={1:.2f}".format(r,p))
        ax.set_xlim(center[i] - 5 * mad[i], center[i] + 5 * mad[i])
        ax.set_ylim(center[j] - 5 * mad[j], center[j] + 5 * mad[j])
        plt.legend(loc=2,prop={'size':10})
        if j==len(cols)-1:
            ax.set_xlabel("{0}".format(labels[i]))
        else:
            ax.xaxis.set_ticklabels([])
        if i==0:
            ax.set_ylabel("{0}".format(labels[j]))
        else:
            ax.yaxis.set_ticklabels([])
        plt.locator_params(nbins=7)

    plt.savefig("figs/index_correlations.png")


