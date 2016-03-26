# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 10:24:35 2013

@author: cbarbosa

Program to produce plots of Lick indices in 1D, comparing with results from 
Coccato et al. 2011
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import  LinearNDInterpolator as interpolator
from matplotlib.colors import Normalize
from scipy import ndimage
import brewer2mpl


from config import *
from mcmc_model import get_model_lims
import newcolorbars as nc

class Ssp:
    """ Wrapper for the interpolated model."""
    def __init__(self, model_table, indices=np.arange(25)):
        self.interpolate(model_table)
        self.indices = indices

    def interpolate(self, model_table):
        modeldata = np.loadtxt(model_table, dtype=np.double)
        self.model = interpolator(modeldata[:,:3], modeldata[:,3:])

    def fn(self, age, metallicity, alpha):
        return self.model(age, metallicity, alpha)[self.indices]

    def __call__(self, pars):
        return self.fn(*pars)

def get_model_range(table):
    """ Get the range for the indices according to models. """
    modeldata = np.loadtxt(table)
    indices = modeldata[:,3:].T
    ranges = np.zeros((len(indices), 2))
    for i, index in enumerate(indices):
        ranges[i] = [index.min(), index.max()]
    return ranges

def line(x, zp, grad ):
    return zp + grad * x

def mask_slits():
    data = np.loadtxt("results.tab", dtype=str)
    mask = ["inn1_s22", "inn1_s25", "inn1_s27", "out1_s19", "out1_s20",
             "out1_s21", "out1_s22","out1_s23", "out1_s24", "out1_s25",
             "out1_s26", "inn2_s39", "cen1_s14", "cen2_s15", "out2_s22",
            "out2_s29", ]
    # mask = ["inn1_s22", "inn1_s25", "inn1_s27", "out1_s19", "out1_s20",
    #         "out1_s21", "out1_s22","out1_s23", "out1_s24", "out1_s25",
    #         "out1_s26", "inn2_s39", "cen1_s14", "cen2_s15", "inn2_s34",
    #         "out1_s18", "cen1_s35", "cen2_s23", ]
    mask = np.array(["fin1_n3311{0}.fits".format(x) for x in mask])
    mask = data[~np.in1d(data[:,0], mask)]
    np.savetxt("results_masked.tab", mask, fmt="%s")
    return "results_masked.tab"

def movingrms(x, y, window_size=10):
    a = np.column_stack((x,y))
    a = a[np.argsort(a[:,0])]
    window = np.ones(window_size)/float(window_size)
    rms = np.sqrt(np.convolve(a[:,1], window, 'same'))
    rms = ndimage.filters.gaussian_filter(rms, 2.5)
    b = np.column_stack((a[:,0], rms))
    b = b[~np.isnan(rms)]
    return b
          
if __name__ == "__main__":
    model_table = os.path.join(tables_dir, "models_thomas_2010.dat")
    ssp = Ssp(model_table)
    restrict_pa = 0
    log = True
    pc = 1
    r_tran = np.log10( 8.4 / re)
    plt.ioff()
    model_table = os.path.join(tables_dir, "models_thomas_2010.dat")
    ranges = get_model_range(model_table)
    ii = [12,13,16,17,18,19]
    ranges = np.array([ranges[i] for i in ii ])
    os.chdir(os.path.join(home, "single2"))
    indices = [r"H$\beta$ [$\AA$]", r"Fe5015 [$\AA$]", r"Mg $b$ [$\AA$]", 
               r"Fe5270 [$\AA$]",r"Fe5335 [$\AA$]",r"Fe5406 [$\AA$]",
               r"Fe5709 [$\AA$]"]
    lodo_table = os.path.join(tables_dir, "coccato2011_indices.tsv")
    lodo = np.loadtxt(lodo_table, usecols=(1,3,5,7,9,11,13,15))
    lodoerr = np.loadtxt(lodo_table, usecols=(1,4,6,8,10,12,14,16))
    with open(lodo_table) as f:
        header = f.readline() [:-1]
    # Converting radius to effective units
    lodo[:,0] /= (4.125 * re)
    if log:
        lodo[:,0] = np.log10(lodo[:,0])
    #############################
    # Applying offsets from paper
    lodo[:,1] += 0.11
    lodo[:,3] += 0.13
    #############################
    # Calculating composite indices for Lodo's data  
    fe5270 = lodo[:,3]
    fe5270_e = lodoerr[:,3]
    fe5335 = lodo[:,4]
    fe5335_e = lodoerr[:,4]
    mgb = lodo[:,2]
    mgb_e = lodoerr[:,2]
    meanfe = 0.5 * (fe5270 + fe5335)
    meanfeerr = 0.5 * np.sqrt(fe5270_e**2 + fe5335_e**2)
    term = (0.72 * fe5270 + 0.28 * fe5335)
    mgfeprime = np.sqrt(mgb *  term)
    mgfeprimeerr = 0.5 * np.sqrt(term /  mgb * (mgb_e**2) +
    mgb / term * ((0.72 * fe5270_e)**2 + (0.28 * fe5335_e)**2))
    lodo2 = np.column_stack((lodo[:,0], lodo[:,1],  lodo[:,3], meanfe, 
                             mgfeprime))
    lodo2err = np.column_stack((lodo[:,0], lodoerr[:,1], lodoerr[:,3],
                                meanfeerr, mgfeprimeerr))
    objs = np.loadtxt(lodo_table, dtype = str, usecols=(0,))
    lododata = np.loadtxt(lodo_table, usecols=np.arange(1,17))
    outtable = np.column_stack((lododata, meanfe, meanfeerr, 
                                mgfeprime, mgfeprimeerr))
    outtable = np.around(outtable, decimals=4)
    outtable = np.column_stack((objs, outtable))
    header += "\t<Fe>\terr\t[MgFe]'\terr\n"
    with open(os.path.join(tables_dir, "coccato2011.dat"), "w") as f:
        f.write(header)
        np.savetxt(f, outtable, fmt="%s")
    ################################
    dwarf = lodo[-1]
    lodo = lodo[:-1]
    dwarferr = lodoerr[-1]
    lodoerr = lodoerr[:-1]
    dwarf2 = lodo2[-1]
    lodo2 = lodo2[:-1]
    dwarf2err = lodo2err[-1]
    lodo2err = lodo2err[:-1]
    ##########################################################################
    # Central values according to Loubser+ 2009
    loubser = np.array([1.581, 5.03, 4.608, 2.773, 2.473, 1.532, 0.876])
    loubser_err = np.array([0.111, 0.228, 0.091, 0.088, 0.099, 0.072, 0.05])
    ##########################################################################
    # Data from Loubser + 2012
    loubser12 = np.loadtxt("/home/kadu/Dropbox/hydra1/loubser2012/"
                           "lick_loubser2012.txt",
                           usecols=(0,13,14,17,18,19,20,21))
    loubser12[:,0] += np.log10(26.6/re) #Scaling to our effective radius
    loubser12_errs = np.loadtxt("/home/kadu/Dropbox/hydra1/loubser2012/"
                                "lick_loubser2012_errs.txt",
                           usecols=(0,13,14,17,18,19,20,21))
    ##########################################################################
    # Mask table
    results_masked = mask_slits()
    ##########################################################################
    # Read data
    r, pa, sn, mu = np.loadtxt(results_masked, usecols=(3,4,14,82)).T
    r /= re # Normalization to effective radius
    if log:
        r = np.log10(r)
    lick = np.loadtxt(results_masked, usecols=(39,41,47,49,51,53,55))
    lickerr = np.loadtxt(results_masked, usecols=(40,42,48,50,52,54,56))
    if restrict_pa:
        good_pa = np.logical_or(np.logical_and(pa > 48, pa < 78), r < r_tran)
        r = r[good_pa]
        lick = lick[good_pa]
        lickerr = lickerr[good_pa]
        sn = sn[good_pa]
    r = r[sn > sn_cut]
    lick = np.transpose(lick[sn > sn_cut])
    lickerr = np.transpose(lickerr[sn > sn_cut])
    #########################################################################
    # Bin data for gradients
    if log:
        rbinnum, redges = np.histogram(r, bins=8, range=(r_tran,r.max()))
    else:
        rbinnum, redges = np.histogram(r, bins=8, range=(10**(r_tran),10**.8))
    data_r = []
    rbins = []
    errs_r = []
    lick_masked = np.ma.array(lick, mask=np.isnan(lick))
    lickerrs_masked = np.ma.array(lickerr, mask=np.isnan(lick))
    for i, bin in enumerate(rbinnum):
        idx = np.logical_and(r >= redges[i], r < redges[i+1])
        if not len(np.where(idx)[0]):
            continue
        median = True
        if median:
            m = np.ma.median(lick_masked[:,idx].T, axis=0) # median
            data_r.append(m)
            rbins.append(np.ma.median(r[idx], axis=0))
        else:
            data_r.append(np.ma.average(lick_masked[:,idx].T, axis=0,
                                     weights=np.power(10, -0.4*mu[idx])))
            rbins.append(np.ma.average(r[idx], axis=0,
                                     weights=np.power(10, -0.4*mu[idx])))
        sigma_mad = 1.4826 * np.ma.median(np.abs(lick_masked[:,idx].T - m),
                                          axis=0)
        sigma =  np.ma.std(lick_masked[:,idx].T, axis=0)
        errs_r.append(sigma_mad)

    data_r = np.array(data_r)
    rbins = np.array(rbins)
    errs_r = np.array(errs_r)
    #########################################################################
    # Taking only inner region for gradients in NGC 3311
    if log:
        idx3311 = np.where(r <= r_tran)[0]
        idxhalo = np.where(r > r_tran)[0]
    else:
        idx3311 = np.where(r <= 10**(r_tran))[0]
        idxhalo = np.where(r > 10**(r_tran))[0]
    r3311 = r[idx3311]
    rhalo = r[idxhalo]
    lick3311 = lick[:,idx3311]
    lickhalo = lick[:,idxhalo]
    errs1_3311 = lickerr[:,idx3311]
    errs_halo = lickerr[:,idxhalo]
    #########################################################################
    # First figure, simple indices
    app = "_pa" if restrict_pa else ""
    mkfig1 = True
    gray = "0.75"
    ##########################################################################
    lims, ranges = get_model_lims(os.path.join(tables_dir,
                          "models_thomas_2010_metal_extrapolated.dat"))
    idx = np.array([12,13,16,17,18,19,20])
    lims = lims[idx]
    # Setting the colormap properties for the scatter plots
    cmap = brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colormap
    cmap = nc.cmap_discretize(cmap, 3)
    color = cm.get_cmap(cmap)
    norm = Normalize(vmin=0, vmax=45)
    if mkfig1:
        plt.figure(1, figsize = (6, 14  ))
        gs = gridspec.GridSpec(7,1)
        gs.update(left=0.15, right=0.95, bottom = 0.1, top=0.94, wspace=0.1,
                  hspace=0.09)
        tex = []
        for j, ll in enumerate(lick):
            # print indices[j], ranges[j], ssp.fn(9.,0.12,.4)[ii[j]]
            if j == 0:
                labels = ["This work", "Coccato et al. 2011",
                          "This work (binned)"]
            else:
                labels = [None, None, None]
            notnans = ~np.isnan(ll)
            ax = plt.subplot(gs[j])
            ydata = ll[notnans]
            ax.errorbar(r[notnans], ydata, yerr=lickerr[j][notnans],
                        fmt=None, color=gray, ecolor=gray, capsize=0, mec=gray,
                        ms=5.5, alpha=1, markerfacecolor="none",
                        mew=2, elinewidth=1 )
            ax.scatter(r[notnans], ydata, c=sn[notnans], s=60, cmap=cmap, zorder=2,
                        lw=0.5, norm=norm, edgecolor="k")
            ax.plot(1000, 1000, "o", mew=0.8, mec="k", c=color(0),
                    label=r"S/N $< 15$")
            ax.plot(1000, 1000, "o", mew=0.8, mec="k", c=color(0.5),
                    label=r"$15\leq$ S/N $\leq 30$")
            ax.plot(1000, 1000, "o", mew=0.8, c=color(1.), mec="k",
                    label=r"S/N $> 30$")
            ax.errorbar(loubser12[:,0], loubser12[:,j+1],
                        yerr=loubser12_errs[:,j+1], color="r", ecolor="r",
                        fmt="s", mec="k", capsize=0, lw=0.2,
                        label= "Loubser et al. 2012", alpha=1, ms=7.5, mew=0.5)
            ax.errorbar(lodo[:,0], lodo[:,j+1],
                         yerr = lodoerr[:,j+1],
                         fmt="^", c="orange", capsize=0, mec="k", ecolor="0.5",
                         label=labels[1], ms=8., alpha=1, lw=0.5, mew=0.5)
            ax.errorbar(dwarf[0],
                        dwarf[j+1], yerr=dwarferr[j+1], fmt="^", c="orange",
                        capsize=0, mec="k", ecolor="0.5", ms=8., lw=0.5, mew=0.5)
            plt.minorticks_on()
            if j+1 != len(lick):
                ax.xaxis.set_ticklabels([])
            else:
                plt.xlabel(r"$\log$ R / R$_{\mbox{e}}$")
            plt.ylabel(indices[j])
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            if j == 0:
                leg = ax.legend(prop={'size':11}, loc=2, ncol=2, fontsize=14,
                                scatterpoints = 1, frameon=False)
            add = 0 if j != 0 else 2
            sigma_mad = 1.48 * np.median(np.abs(ydata - np.median(ydata)))
            ym = np.ceil(np.median(ydata)-4 * sigma_mad)
            yp = np.floor(np.median(ydata)+4*sigma_mad+add)
            ylim = plt.ylim(ym, yp)
            ##################################################################
            # Measuring gradients
            ##################################################################
            # NGC 3311
            l = lick3311[j]
            lerr = errs1_3311[j]
            mask = ~np.isnan(l)
            popt, pcov = curve_fit(line, r3311[mask], l[mask], sigma=lerr[mask])
            pcov = np.sqrt(np.diagonal(pcov))
            x = np.linspace(r.min(), r_tran, 100)
            if not log:
                x = 10**x
            y = line(x, popt[0], popt[1])
            lll, = ax.plot(x, y, "--k", lw=2, zorder=10000)
            lll.set_dashes([10, 3])
            # Including shades for +- 1%'
            ##################################################################
            # Halo
            values = lickhalo[j]
            for k,v in enumerate(values):
                if v <= lims[j][0] or v >= lims[j][1]:
                    values[k] = np.nan
            mask = ~np.isnan(values)
            l = lickhalo[j][mask]
            lerr = errs_halo[j][mask]
            popth, pcovh = curve_fit(line, rhalo[mask], l, sigma=lerr)
            pcovh = np.sqrt(np.diagonal(pcovh))
            x = np.linspace(r_tran, 0.7, 100)
            if not log:
                x = 10**x
            y = line(x, popth[0], popth[1])
            lll, = ax.plot(x, y, "-k", lw=1.5, zorder=10000)
            lll.set_dashes([10, 3])
            #################################################################
            # Ploting rms 1%
            for p, c in [[1,"0.3"]]:
                tab = os.path.join(tables_dir,
                                        "rms_{1}pc_lick_{0}.txt".format(j, p))
                print tab
                rms = np.loadtxt(os.path.join(tables_dir,
                                        "rms_{1}pc_lick_{0}.txt".format(j, p)),
                                 usecols=(0,1))
                xrms, yrms = rms[rms[:,0] < r_tran].T
                # ax.plot(xrms, yrms + line(xrms, popt[0], popt[1]), "-", c="0.5")
                # ax.plot(xrms, -yrms + line(xrms, popt[0], popt[1]), "-", c="0.5")
                ax.fill_between(xrms, yrms + line(xrms, popt[0], popt[1]),
                                line(xrms, popt[0], popt[1]) - yrms,
                                edgecolor="none", color=gray,
                                linewidth=0, alpha=1)
            ##################################################################
            # Outer halo in bins
            # popt2, pcov2 = curve_fit(line, rbins, data_r[:,j], sigma=errs_r[:,j])
            # pcov2 = np.sqrt(np.diagonal(pcov2))
            # x = np.linspace(r_tran, r.max(), 100)
            # if not log:
            #     x = 10**x
            # y = line(x, popt2[0], popt2[1])
            # ax.plot(x, y, "--k", lw=2)
            # ax.axvline(x=r_tran, c="k", ls="-.")
            ##################################################################
            # Ploting rms 1%
            # for p, c in [[1,"0.1"], [6, "0.8"]]:
            for p, c in [[1,"0.1"]]:
                rms = np.loadtxt(os.path.join(tables_dir,
                                        "rms_{1}pc_lick_{0}.txt".format(j, p)),
                                 usecols=(0,1))
                xrms, yrms = rms[rms[:,0]>=r_tran].T
                ax.fill_between(xrms, yrms + line(xrms, popth[0], popth[1]),
                                line(xrms, popth[0], popth[1]) - yrms,
                                edgecolor="none", color=gray,
                                linewidth=0, alpha=1)
            ##################################################################
            # Draw arrows to indicate central limits
            ax.annotate("", xy=(-1.12, loubser[j]), xycoords='data',
            xytext=(-1.3, loubser[j]), textcoords='data',
            arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", ec="r",
                            lw=2))
            ##################################################################
            ax.set_xlim(-1.35, 1.)
            # ##################################################################
            tex.append(r"{0} & {1[0]:.1f}$\pm${2[0]:.1f} & {1[1]:.1f}$\pm${2[1]:.1f}" \
            r" & {3[0]:.1f}$\pm${4[0]:.1f} & {3[1]:.1f}$\pm${4[1]:.1f}""\\\\".format(
                indices[j][:-7], popt, pcov, popth, pcovh))
            print indices[j][:-7],
            for m in [1,2,3]:
                print np.abs(popt[1] - popth[1]) < m * (pcov[1]+pcovh[1]),
            print
        print "Saving new figure..."
        plt.savefig("figs/lick_radius.png".format(pc), dpi=300,
                    bbox_inches="tight", transparent=False)
        for t in tex:
            print t
    # plt.show(block=1)
    # Making plots of Hbeta, Mgb, <Fe> and [MgFe]'
    # r, pa, sn = np.loadtxt(results_masked, usecols=(3,4,14)).T
    # # r /= re
    # lick = np.loadtxt(results_masked, usecols=(39, 67, 80))
    # lickerr = np.loadtxt(results_masked, usecols=(40, 68, 81))
    # if restrict_pa:
    #     good_pa = np.logical_and(pa > 0, pa < 270)
    #     r = r[good_pa]
    #     lick = lick[good_pa]
    #     lickerr = lickerr[good_pa]
    #     sn = sn[good_pa]
    # r = r[sn > sn_cut]
    # lick = np.transpose(lick[sn > sn_cut])
    # lickerr = np.transpose(lickerr[sn > sn_cut])
    # gs2 = gridspec.GridSpec(len(lick),3)
    # gs2.update(left=0.15, right=0.95, bottom = 0.1, top=0.94, hspace = 0.10,
    #            wspace=0.04)
    # plt.figure(2, figsize = (6, 7))
    # indices = [r"H$\beta$ [$\AA$]", r"[MgFe]'",
    #            r"$\mbox{Mg }b/\langle\mbox{Fe}\rangle$"]
    #
    # for j, (ll,lerr) in enumerate(zip(lick, lickerr)):
    #     ax = plt.subplot(gs2[j, 0:2], xscale="log")
    #     notnans = ~np.isnan(ll)
    #     ax.errorbar(r[notnans], ll[notnans], yerr=lerr[notnans],
    #                 fmt="d", color="r",
    #                 ecolor=gray, capsize=0, mec="k", markerfacecolor="none")
    #     # plt.errorbar(lodo2[:,0], lodo2[:,j+1],
    #     #              yerr=lodo2err[:,j+1], fmt="+", c="b", capsize=0,
    #     #              mec="b", ecolor="0.5", label=None, ms=10)
    #     # plt.errorbar(dwarf2[0],
    #     #              dwarf2[j+1], yerr=dwarf2err[j+1], fmt="o",
    #     #              c="w", capsize=0, mec="b", ecolor="0.5")
    #     plt.minorticks_on()
    #     if j != len(lick) -1 :
    #         ax.xaxis.set_ticklabels([])
    #     else:
    #         plt.xlabel(r"R (kpc)")
    #         ax.set_xticklabels(["0.1", "1", "10"])
    #     plt.ylabel(indices[j], fontsize=10)
    #     ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    #     plt.ylim(ylims[j])
    #     # Histograms
    #     ax = plt.subplot(gs2[j, 2])
    #     plt.minorticks_on()
    #     ax.hist(ll[notnans], orientation="horizontal", color="r",
    #             ec="k")
    #     ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    #     ax.xaxis.set_ticklabels([])
    #     ax.yaxis.set_ticklabels([])
    #     plt.ylim(ylims[j])
    # plt.savefig("figs/lick_radius_combined.pdf",
    #             bbox_inches="tight", transparent=False)