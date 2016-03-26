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
from scipy.optimize import fmin, fminbound
from scipy.integrate import quad
import matplotlib.cm as cm
from matplotlib.mlab import normpdf
from matplotlib.colors import Normalize
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.mixture import GMM

from config import *
from run_ppxf import speclist

class Dist():
    """ Simple class to handle the distribution data of MCMC. """
    def __init__(self, data, lims, bw=0.005):
        self.data = data
        self.lims = lims
        self.bw = bw
        self.genextreme = genextreme(self.data)
        self.norm = statdist(self.data, stats.norm, "norm")
        # self.truncnorm = statdist(self.data, stats.truncnorm, "truncnorm")
        # self.gmm = gmm(self.data)
        dists = [self.genextreme, self.norm]
        idx = np.argmin([x.ad for x in dists])
        self.best = dists[idx]
        self.MAPP = self.best.MAPP
        self.calc_err()
        # # Calculate percentiles
        # self.percentileatmapp =  stats.percentileofscore(self.data, self.MAPP)
        # self.percentilemax = np.minimum(self.percentileatmapp + 34., 100.)
        # self.percentilemin = np.maximum(self.percentileatmapp - 34., 0.)
        # self.MAPPmin = stats.scoreatpercentile(self.data, self.percentilemin)
        # self.MAPPmax = stats.scoreatpercentile(self.data, self.percentilemax)
        self.lerr = np.sqrt(self.bw**2 + np.abs(self.MAPP - self.MAPPmin)**2)
        self.uerr = np.sqrt(self.bw**2 + np.abs(self.MAPP - self.MAPPmax)**2)
        return

    def calc_err(self):
        """ Calculate error for the best distribution. """
        r = np.abs(self.lims[0] - self.lims[1])
        def integral(y, return_x=False):

            x0 = float(fminbound(lambda x: np.abs(self.best.pdf(x) - y),
                       self.lims[0], self.best.MAPP, full_output=1)[0])
            x1 = float(fminbound(lambda x: np.abs(self.best.pdf(x) - y),
                       self.best.MAPP, self.lims[1], full_output=1)[0])
            if not return_x:
                return quad(self.best.pdf, x0, x1)[0]
            else:
                return x0, x1
        y = fmin(lambda x: np.abs(integral(x) - 0.68),
                        0.6 * self.best.pdf(self.best.MAPP), disp=0)
        self.MAPPmin,  self.MAPPmax = integral(y, return_x=True)
        return


class genextreme():
    def __init__(self, data):
        self.dist = stats.genextreme
        self.distname = "genextreme"
        self.data = data
        self.p = self.dist.fit(self.data)
        self.frozen = self.dist(self.p[0], loc=self.p[1], scale=self.p[2])
        self.pdf = lambda x : self.frozen.pdf(x)
        self.sample = self.frozen.rvs(len(self.data))
        self.sample2 = self.frozen.rvs(100000)
        self.moments = self.frozen.stats(moments="mvsk")
        self.MAPP = fmin(lambda x: -self.pdf(x),
                        self.moments[0], disp=0)[0]
        try:
            self.ad =  stats.anderson_ksamp([self.sample, self.data])[0]
        except:
            self.ad = np.infty

class statdist():
    def __init__(self, data, dist, distname):
        self.dist = dist
        self.distname = distname
        self.data = data
        self.p = self.dist.fit(self.data)
        self.pdf = lambda x : self.dist.pdf(x, *self.p[:-2], loc=self.p[-2],
                                            scale=self.p[-1])
        self.sample = stats.norm.rvs(self.p[0], size=len(self.data),
                                           scale=self.p[-1])
        self.moments = self.dist.stats(*self.p, moments="mvsk")
        self.MAPP = fmin(lambda x: -self.pdf(x),
                        self.moments[0], disp=0)[0]
        try:
            self.ad =  stats.anderson_ksamp([self.sample, self.data])[0]
        except:
            self.ad = np.infty

class gmm():
    def __init__(self, data):
        self.distname = "gmm"
        self.data = data
        self.n_components = np.arange(1,11)
        self.models = []
        self.X = np.reshape(self.data, (len(self.data),1))
        for i in self.n_components:
            self.models.append(GMM(i, covariance_type='full').fit(self.X))
        self.AIC = np.array([m.aic(self.X) for m in self.models])
        self.BIC = np.array([m.bic(self.X) for m in self.models])
        self.k = 2 * np.arange(1,11)
        self.n = len(self.data)
        self.AICc = self.AIC + 2*self.k * (self.k + 1) / (self.n - self.k - 1)
        self.imin = np.minimum(np.argmin(self.AIC), np.argmin(self.BIC))
        self.best = self.models[self.imin]

def hist2D(dist1, dist2, ax):
    """ Plot distribution and confidence contours. """
    X, Y = np.mgrid[dist1.lims[0] : dist1.lims[1] : 20j,
                    dist2.lims[0] : dist2.lims[1] : 20j]
    extent = [dist1.lims[0], dist1.lims[1], dist2.lims[0], dist2.lims[1]]
    positions = np.vstack([X.ravel(), Y.ravel()])    
    values = np.vstack([dist1.data, dist2.data])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    ax.imshow(np.rot90(Z), cmap="gray_r", extent=extent, aspect="auto",
              interpolation="spline16")
    # plt.hist2d(dist1.data, dist2.data, bins=40, cmap="gray_r")
    plt.axvline(dist1.MAPP, c="r", ls="--", lw=1.5)
    plt.axhline(dist2.MAPP, c="r", ls="--", lw=1.5)
    plt.tick_params(labelsize=10)
    ax.minorticks_on()
    plt.locator_params(axis='x',nbins=10)
    return

def summary_table(specs, modelname, db):
    """ Make final table."""
    lines = []
    for spec in specs:
        folder = spec.replace(".fits", "_db{0}".format(db))
        logfile = os.path.join(working_dir, folder,
                               "summary.txt")
        if not os.path.exists(logfile):
            continue
        with open(logfile, "r") as f:
            header = f.readline()
            lines.append(f.readline())
    table = os.path.join(working_dir, "populations_{0}.txt".format(modelname))
    with open(table, "w") as f:
        f.write(header)
        f.write("\n".join(lines))

if __name__ == "__main__":
    working_dir = os.path.join(home, "single2")
    os.chdir(working_dir)
    plt.ioff()
    specs = speclist()
    specs = ["fin1_n3311cen1_s23.fits", "fin1_n3311cen1_s30.fits"]
    db = ""
    modelname = "miles" if db == "2" else "thomas"
    dirs = [x.replace(".fits", "_db{0}".format(db)) for x in specs]
    lims = [[9 + np.log10(1.), 9 + np.log10(15.)], [-2.25, 0.90], [-0.3, 0.5]]
    plims = [[np.log10(1.), 1.2], [-2.3, 0.7], [-0.4, 0.6]]
    fignums = [4, 7, 8]
    pairs = [[0,1], [0,2], [1,2]]
    plt.ioff()
    pp = PdfPages(os.path.join(working_dir,
                               "mcmc_results_{0}.pdf".format(modelname)))
    plt.figure(1, figsize=(9,6.5))
    plt.minorticks_on()
    table_summary, table_results = [], []
    sndata = dict(np.loadtxt("ppxf_results.dat", usecols=(0,10), dtype=str))
    for spec in specs:
        print spec
        # continue
        folder = spec.replace(".fits", "_db{0}".format(db))
        if not os.path.exists(os.path.join(working_dir, folder)):
            continue
        os.chdir(os.path.join(working_dir, folder))
        name = spec.replace(".fits", '').replace("n3311", "").split("_")
        name = name[1] + name[2]
        name = r"{0}".format(name)
        sn = float(sndata[spec])
        ages_data = np.loadtxt("Chain_0/age_dist.txt")
        ages_data = 9. + np.log10(ages_data)
        ages = Dist(ages_data, [9 + np.log10(1), 9 + np.log10(15)])
        metal_data = np.loadtxt("Chain_0/metal_dist.txt")
        metal = Dist(metal_data, [-2.25, 0.90])
        alpha_data = np.loadtxt("Chain_0/alpha_dist.txt")
        alpha = Dist(alpha_data, [-0.3, 0.5])
        dists = [ages, metal, alpha]
        log, summary = [r"{0:28s}".format(spec)], []
        for i, d in enumerate(dists):
            weights = np.ones_like(d.data)/len(d.data)
            ax = plt.subplot(3,3,(4*i)+1)
            # plt.tick_params(labelsize=10)
            N, bins, patches = plt.hist(d.data, color="b",ec="k", bins=30,
                range=tuple(lims[i]), normed=True, edgecolor="k",
                                        histtype='bar',linewidth=1.)
            fracs = N.astype(float)/N.max()
            norm = Normalize(-.2* fracs.max(), 1.5 * fracs.max())
            for thisfrac, thispatch in zip(fracs, patches):
                color = cm.gray_r(norm(thisfrac))
                thispatch.set_facecolor(color)
                thispatch.set_edgecolor("w")
            x = np.linspace(d.data.min(), d.data.max(), 100)
            tot = np.zeros_like(x)
            # for m,w,c in zip(d.gmm.best.means_, d.gmm.best.weights_,
            #                  d.gmm.best.covars_):
            #     y = w * normpdf(x, m, np.sqrt(c))[0]
            #     ax.plot(x, y, "--b")
            #     tot += y
            # ax.plot(x,tot, "-b", lw=2)
            # pdf = np.exp(logprob)
            # pdf_individual = responsibilities * pdf[:, np.newaxis]
            # print pdf_individual
            ylim = ax.get_ylim()
            plt.plot(x, d.best.pdf(x), "-r", label="AD = {0:.1f}".format(
                d.best.ad), lw=1.5, alpha=0.7)
            ax.set_ylim(ylim)
            # plt.legend(loc=2, prop={'size':8})
            plt.axvline(d.best.MAPP, c="r", ls="--", lw=1.5)
            plt.tick_params(labelright=True, labelleft=False, labelsize=10)
            plt.xlim(d.lims)
            plt.locator_params(axis='x',nbins=10)
            if i < 2:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                plt.xlabel(r"[$\mathregular{\alpha}$ / Fe]")
            plt.minorticks_on()
            summary.append([d.best.MAPP, d.uerr, d.lerr])
            for ss in [d.MAPP, d.MAPPmin, d.MAPPmax, d.best.ad]:
                log.append(r"{0:10s}".format(r"{0:.5f}".format(ss)))
        logfile = os.path.join(working_dir, folder,
                               "summary.txt".format(modelname))
        with open(logfile, "w") as f:
            f.write("{0:28s}{1:10s}{2:10s}{3:10s}{6:10s}{4:10s}{2:10s}{3:10s}{6:10s}{5:10s}"
                "{2:10s}{3:10s}{6:10s}\n".format("#Spectra", "Log AGE", "LOWER",
                "UPPER", "[Z/H]", "[E/Fe]", "AD test"))
            f.write("".join(log))
        ax = plt.subplot(3,3,4)
        hist2D(ages, metal, ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.ylabel("[Z/H]")
        
        ax = plt.subplot(3,3,7)
        hist2D(ages, alpha, ax)
        plt.ylabel(r"[$\mathregular{\alpha}$ / Fe]")
        plt.xlabel("log Age (yr)")
        ax = plt.subplot(3,3,8)
        plt.xlabel("[Z/H]")
        hist2D(metal, alpha, ax)
        # Annotations
        plt.annotate(r"Spectrum: {0}    S/N={1:.1f}".format(name.upper(), sn),
                     xy=(.7,.91),
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
                    "logs/mcmc_{0}_{1}.png".format(name, modelname)), dpi=300)
        plt.clf()
    pp.close()
    summary_table(speclist(), modelname, db)

        
        