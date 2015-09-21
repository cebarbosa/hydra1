# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 17:29:58 2013

@author: cbarbosa

Program to verify results from MCMC runs. 

"""

import os

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import fmin
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages

from config import *
from run_ppxf import speclist

class Dist():
    """ Simple class to handle the distribution data of MCMC. """
    def __init__(self, data, lims, bw=0.005):
        self.data = data
        self.lims = lims
        self.bw = bw
        self.kernel = stats.gaussian_kde(self.data, bw_method=self.bw)
        self.skew, self.pvalue = stats.skewtest(self.data)
        self.param = stats.genextreme.fit(self.data)
        self.dist = stats.genextreme
        self.pdf = lambda x : self.dist.pdf(x,
                            *self.param[:-2], loc=self.param[-2],
                                    scale=self.param[-1])
        self.MAPP = fmin(lambda x: -self.pdf(x),
                         0.5 * (self.data.min() + self.data.max()), disp=0)[0]
        # Calculate percentiles
        self.percentileatmapp =  stats.percentileofscore(self.data, self.MAPP)
        self.percentilemax = np.minimum(self.percentileatmapp + 34., 100.)
        self.percentilemin = np.maximum(self.percentileatmapp - 34., 0.)
        self.MAPPmin = stats.scoreatpercentile(self.data, self.percentilemin)
        self.MAPPmax = stats.scoreatpercentile(self.data, self.percentilemax)
        self.lerr = np.sqrt(self.bw**2 + np.abs(self.MAPP - self.MAPPmin)**2)
        self.uerr = np.sqrt(self.bw**2 + np.abs(self.MAPP - self.MAPPmax)**2)
        return


def hist2D(dist1, dist2):
    """ Plot distribution and confidence contours. """
    X, Y = np.mgrid[dist1.lims[0] : dist1.lims[1] : 20j, 
                    dist2.lims[0] : dist2.lims[1] : 20j]
    extent = [dist1.lims[0], dist1.lims[1], dist2.lims[0], dist2.lims[1]]
    positions = np.vstack([X.ravel(), Y.ravel()])    
    values = np.vstack([dist1.data, dist2.data]) 
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)    
    ax.imshow(np.rot90(Z), cmap="gray_r", extent= extent, aspect="auto")
    pers = []
    for per in np.array([32, 5, 0.3]):
        pers.append(stats.scoreatpercentile(kernel(kernel.resample(1000)), 
                                            per))
    plt.contour(Z.T, np.array(pers), colors="k", extent=extent)
    plt.axvline(dist1.MAPP, c="r", ls="--")
    plt.axhline(dist2.MAPP, c="r", ls="--")
    plt.minorticks_on()
    return

if __name__ == "__main__":
    working_dir = os.path.join(home, "single2")
    os.chdir(working_dir)
    plt.ioff()
    specs = speclist()
    dirs = [x.replace(".fits", "_db") for x in specs]
    lims = [[np.log10(1.), np.log10(15.)], [-2.25, 0.67], [-0.3, 0.5]]
    plims = [[np.log10(1.), 1.2], [-2.3, 0.7], [-0.4, 0.6]]
    fignums = [4, 7, 8]
    pairs = [[0,1], [0,2], [1,2]]
    plt.ioff()
    pp = PdfPages(os.path.join(working_dir, "mcmc_results.pdf"))
    plt.figure(1, figsize=(9,6.5))
    plt.minorticks_on()
    table_summary, table_results = [], []
    for spec in specs:
        print spec
        folder = spec.replace(".fits", "_db")
        if not os.path.exists(os.path.join(working_dir, folder)):
            continue
        os.chdir(os.path.join(working_dir, folder))
        name = spec.replace(".fits", '').replace("n3311", "").split("_")
        name = name[1] + name[2]
        name = r"{0}".format(name)
        ages_data = np.loadtxt("Chain_0/age_dist.txt")
        ages_data = np.log10(ages_data)
        ages = Dist(ages_data, [np.log10(1),np.log10(15)])
        metal_data = np.loadtxt("Chain_0/metal_dist.txt")
        metal = Dist(metal_data, [-2.25, 0.67])
        alpha_data = np.loadtxt("Chain_0/alpha_dist.txt")
        alpha = Dist(alpha_data, [-0.3, 0.5])
        weights = np.ones_like(ages.data)/len(ages.data)
        dists = [ages, metal, alpha]
        table1, table2, summary = [], [], []
        for i, d in enumerate(dists):
            ax = plt.subplot(3,3,(4*i)+1)
            N, bins, patches = plt.hist(d.data, color="w", weights=weights,
                                        ec="w", bins=35, range=tuple(lims[i]),
                                        normed=True)
            fracs = N.astype(float)/N.max()
            norm = Normalize(-.2* fracs.max(), 1.5 * fracs.max())
            for thisfrac, thispatch in zip(fracs, patches):
                color = cm.gray_r(norm(thisfrac))
                thispatch.set_facecolor(color)
            x = np.linspace(d.data.min(), d.data.max(), 100)
            pdf_fitted = d.dist.pdf(x, *d.param[:-2], loc=d.param[-2],
                                    scale=d.param[-1])
            plt.plot(x, pdf_fitted, "-r")
            plt.axvline(d.MAPP, c="r", ls="--")
            plt.tick_params(labelright=True, labelleft=False)
            plt.xlim(d.lims)
            if i < 2:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel(r"[$\mathregular{\alpha}$ / Fe]")
            plt.minorticks_on()
            summary.append([d.MAPP, d.uerr, d.lerr])
            table1.append(r"{0:.5f}".format(d.MAPP))
            table1.append(r"{0:.5f}".format(d.MAPPmin))
            table1.append(r"{0:.5f}".format(d.MAPPmax))
            table2.append(r"{0:.5f}".format(d.MAPP))
            table2.append(r"{0:.5f}".format(d.lerr))
            table2.append(r"{0:.5f}".format(d.uerr))
        table1 = [r"{0:28s}".format(spec)] + [r"{0:10s}".format(x) for x in
                                              table1]
        table_results.append("".join(table1))
        table2 = [r"{0:28s}".format(spec)] + [r"{0:10s}".format(x) for x in
                                              table2]
        table_summary.append("".join(table2))
        ax = plt.subplot(3,3,4)
        hist2D(ages, metal)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.ylabel("[Z/Fe]")
        
        ax = plt.subplot(3,3,7)
        hist2D(ages, alpha) 
        plt.ylabel(r"[$\mathregular{\alpha}$ / Fe]")
        plt.xlabel("log Age (Gyr)")
        ax = plt.subplot(3,3,8)
        plt.xlabel("[Z/Fe]")
        hist2D(metal, alpha)
        # Annotations
        plt.annotate(r"Spectrum: {0}".format(name.upper()), xy=(.7,.91),
                     xycoords="figure fraction", ha="center", size=20)
        xys = [(.7,.84), (.7,.77), (.7,.70)]
        line = r"{0:28s}".format(spec)
        for j, par in enumerate([r"Log Age", r"[Z/H]", r"[$\alpha$/Fe]"]):
            text = r"{0}={1[0]:.2f}$^{{+{1[1]:.2f}}}_"" \
                   ""{{-{1[2]:.2f}}}$ dex".format(par, summary[j])
            plt.annotate(text, xy=xys[j], xycoords="figure fraction",
                         ha="center", size=20)
            line += "{0[1]:.5f}"
        plt.tight_layout(pad=0.2)
        # plt.pause(0.001)
        # plt.show(block=True)
        pp.savefig()
        plt.savefig(os.path.join(working_dir,
                    "logs/mcmc_{0}.png".format(name)), dpi=100)
        plt.clf()
    pp.close()
    with open(os.path.join(working_dir, "populations_summary.txt"), "w") as f:
        f.write("{0:28s}{1:10s}{2:10s}{3:10s}{4:10s}{2:10s}{3:10s}{5:10s}"
                "{2:10s}{3:10s}\n".format("#Spectra", "Log AGE", "LERR", "UERR",
                                        "[Z/H]", "[E/Fe]"))
        f.write("\n".join(table_summary))
    with open(os.path.join(working_dir, "populations.txt"), "w") as f:
        f.write("{0:28s}{1:10s}{2:10s}{3:10s}{4:10s}{2:10s}{3:10s}{5:10s}"
                "{2:10s}{3:10s}\n".format("#Spectra", "Log AGE", "LOWER",
                "UPPER", "[Z/H]", "[E/Fe]"))
        f.write("\n".join(table_results))
        
        