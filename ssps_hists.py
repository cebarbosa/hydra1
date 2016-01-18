# -*- coding: utf-8 -*-
"""

Created on 27/10/15

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GMM
from matplotlib.mlab import normpdf
from scipy.stats import ks_2samp, anderson_ksamp

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

def mean_errors(table):
    """ Calculate the maximum error in a given parameter for """
    errp = np.loadtxt(table, usecols=(70,73,76,85)).T
    errm = np.loadtxt(table, usecols=(71,74,77,86)).T
    return np.transpose(0.5 * np.abs(errp - errm))


if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    ###########################################################################
    # Getting data from Spolaor et al. 2010
    tab = os.path.join(tables_dir, "spolaor/c3.txt")
    sp10 = np.loadtxt(tab, usecols=(3,5,7,5)).T
    sp10[0] += 9.
    sp10[3] -= 0.94 * sp10[2]
    ##########################################################################
    # Cols: 0:R, 1:PA, 2-5: SSPs, 6:S/N, 7-10:AD test
    data = np.loadtxt("results_masked.tab", usecols=(3,4,69,72,75,84,14))
    specs = np.loadtxt("results_masked.tab", usecols=(0,), dtype=str)
    sn = np.loadtxt("results_masked.tab", usecols=(14,))
    # Calculating maximum errors
    err = mean_errors("results_masked.tab")
    err_cut = np.array([0.2, 0.7, 0.22, 0.8])
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
    parameters = [r" log Age (Gyr)", "[Z/H]", r"[$\alpha$/Fe]", "[Fe/H]"]
    filenames = ["age_dist.txt", "metal_dist.txt", "alpha_dist.txt",
                 "iron_dist.txt"]
    colors = ["w", "r", "gray"]
    labels = ["All", "Off-Centered \nEnvelope (N=", "Symmetric Halo\n(N="]
    tablab = ["All", "Off-Centered Envelope", "Symmetric Halo"]
    fig = plt.figure(1, figsize=(7,7))
    plt.subplots_adjust(bottom=0.08, right=0.98, top=0.98, left=0.1)
    ylims = [0.5, 0.5, 0.4, 0.5]
    xlims = [[10,10.2], [-2,1], [-.3,0.5], [-1.5,1]]
    for j, (values, error) in enumerate(zip(ssps,err)):
        print parameters[j]
        header = [parameters[j], "N", "MEAN", "SIGMA", "WEIGHT"]
        # print " & ".join(header) + "\\\\"
        vs = []
        for k, i in enumerate([idx, idx_ne, idx_others]):
            e = error[i]
            v = values[i]
            cond = np.where(e < err_cut[j])[0]
            if k == 2:
                np.savetxt(filenames[j].replace(".txt", "_sym.txt"),
                           np.column_stack((specs[i][cond], v[cond])),
                           fmt="%s")
            if k == 0:
                bins= np.histogram(v[cond], bins=11, range=xlims[j], normed=1)[1]
                d = gmm(v[cond])
                d.best = d.models[np.argmin(d.BIC)]
                for l, (m,w,c) in enumerate(zip(d.best.means_, d.best.weights_,
                                 d.best.covars_)):
                    line = [tablab[k], len(v[cond]), round(m, 2),
                                round(np.sqrt(c),2), round(w,2)]
                    # print " & ".join([str(x) for x in line]) + "\\\\"
            else:
                ax = plt.subplot(2,2,j+1)
                plt.locator_params(nbins=5)
                ax.minorticks_on()
                weights = np.ones_like(v[cond])/float(len(v[cond]))
                ax.hist(v[cond], bins=bins, normed=False, weights=weights,
                        alpha=0.5, histtype='stepfilled', color=colors[k],
                        edgecolor="none", label=labels[k]+str(len(v[cond]))+")")
                vs.append(v[cond])
                ax.set_xlim(*xlims[j])
                ax.set_xlabel(parameters[j])
                # ax.axvline(v[cond].mean(), c=colors[k], lw=2, ls="--")
                plt.legend(loc='upper left', prop={'size':10}, frameon=False)
                d = gmm(v[cond])
                imin = np.argmin(d.BIC)
                d.best = d.models[np.argmin(d.BIC)]
                # d.best = d.models[1]
                x = np.linspace(xlims[j][0], xlims[j][1], 100)
                # print np.median(v[cond])
                for l, (m,w,c) in enumerate(zip(d.best.means_, d.best.weights_,
                                 d.best.covars_)):
                    y = w * normpdf(x, m, np.sqrt(c))[0]
                    ax.plot(x, y * (bins[1] - bins[0]), "--", c=colors[k])
                    ax.arrow(float(m), 0, 0, 0.12 * ylims[j],
                             head_width=0.02 * (xlims[j][1] - xlims[j][0]),
                             head_length=0.05 * ylims[j],
                             fc=colors[k], ec=colors[k])
                    # ax.axvline(m, c=colors[k], ls="--", lw=1.5)
                    if l == 0:
                        line = [r"\multirow{{{1}}}{{*}}{{{0}}}".format(parameters[j],
                         len(d.best.means_)), tablab[k], len(v[cond]), round(m, 2),
                                round(np.sqrt(c),2), round(w,2)]
                    else:
                        line = [" ", " ", round(m, 2), round(np.sqrt(c),2),
                                round(w,2)]
                if j in [0,2]:
                    ax.set_ylabel(r"Fraction of total")
                ax.set_ylim(0, ylims[j])
                #     print " & ".join([str(xx) for xx in line]) + "\\\\"
                # print  "\multicolumn{6}{c}{- - - - - -}\\\\"
        vs.append(sp10[j])
        print len(vs[0]), len(vs[1]), len(vs[2])
        print "NE + SYM: ", ks_2samp(vs[0], vs[1])
        print "NE + VIRGO/FORNAX: ", ks_2samp(vs[0], vs[2])
        print "SYM + VIRGO/FORNAX: ", ks_2samp(vs[1], vs[2])
        print "NE + SYM: ", anderson_ksamp((vs[0], vs[1]))
        print  "NE + VIRGO/FORNAX: ", anderson_ksamp((vs[0], vs[2]))
        print "SYM + VIRGO/FORNAX: ", anderson_ksamp((vs[1], vs[2]))
        print
    plt.pause(0.001)
    plt.savefig(os.path.join(os.getcwd(), "figs/hist_outer.png"))
    # plt.show()