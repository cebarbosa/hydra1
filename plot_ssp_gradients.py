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
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import brewer2mpl

from config import *
import newcolorbars as nc

def f(x, zp, grad ):
    return zp + grad * x

def residue(p, x, y, yerr):
    return (p[1] * x +    p[0] - y) / yerr


if __name__ == "__main__":
    restrict_pa = False
    mag_grads = True
    smooth = False
    r_tran = .0
    mu_tran = 22.2
    lum_weight = True
    plt.ion()
    os.chdir(os.path.join(home, "single2"))
    pars = [r"log Age (yr)", r"[Z/H]", r"[$\alpha$/Fe]", r"[Fe/H]"]
    pars2 = [r"log Age (yr)", r"[Z/H]", r"[$\alpha$/Fe]", r"[Fe/H]"]
    table = "results_masked.tab"
    ##########################################################################
    # Read coordinates and S/N
    r, pa, sn, mu = np.loadtxt(table, usecols=(3,4,14,82)).T
    ##########################################################################
    # Rescale to effective units
    r /= re
    r = np.log10(r)
    ##########################################################################
    # Define vector for SPs of Lodo's paper
    ##########################################################################
    # rl = np.array([43.2, 50.8, 61.3, 70.2, 117., 45.4]) / re / 4.125
    rlodo = np.array([42.8, 50.2, 57.1, 67.2, 108.4, 45.]) / re / 4.125
    pal = np.array([47.1, 83.7, 10.3, 102.5, 118.8, 64.])
    rlodo = np.log10(rlodo)
    mulodo = np.array([22.33, 22.57, 22.72, 23.07, 24.13,22.34 ])
    al = np.log10(np.array([13.5, 13.5, 13.5, 13.5, 13.5, 13.5]))
    al += 9.
    ml = np.array([-.46, -.33, -.73, -.39, -.34, -.85])
    alpl = np.array([.45, .48, .5, .44, .5, -.03])
    data_lodo= np.column_stack((al, ml, alpl, ml + 0.94 * alpl)).T
    ##########################################################################
    # Central values according to Loubser+ 2009
    loubser = np.array([np.log10(np.power(10, 0.94) * 10**9),
                        0.12, 0.4, 0.12 -0.94 * 0.4 ])
    ##########################################################################
    # Stellar populations by Loubser + 2012
    l12_table = os.path.join(tables_dir, "loubser12_populations.txt")
    r_l12 = np.loadtxt(l12_table, usecols=(0,)) + np.log10(26.6/re)
    l12 = np.loadtxt(l12_table, usecols=(1,4,7,4)).T
    l12[0] += 9
    l12[3] += -0.94 * l12[2]
    l12_errs1= np.loadtxt(l12_table, usecols=(2,5,8,5)).T
    l12_errs1[0] += 9
    l12_errs1[3] += -0.94 * l12_errs1[2]
    l12_errs2 = np.loadtxt(l12_table, usecols=(3,6,9,6)).T
    l12_errs2[0] += 9
    l12_errs2[3] += -0.94 * l12_errs2[2]
    l12_sb = np.loadtxt(l12_table, usecols=(10,))
    for i in range(3):
        l12_errs1[i] = l12[i] - l12_errs1[i]
        l12_errs2[i] = l12_errs2[i] - l12[i]
    ##########################################################################
    # Read SSP parameters and convert errors for plot
    data = np.loadtxt(table, usecols=(69, 72, 75, 84)).T
    data[0] += 9.
    x, y, sn = np.loadtxt(table, usecols=(1,2,14)).T
    errs1 = np.loadtxt(table, usecols=(70,73,76,85)).T
    errs2 = np.loadtxt(table, usecols=(71,74,77,86)).T
    errs1[0] += 9.
    errs2[0] += 9
    for i in range(4):
        errs1[i] = data[i] - errs1[i]
        errs2[i] = errs2[i] - data[i]
    #########################################################################
    # Taking only inner region for gradients in NGC 3311
    idx3311 = np.where(r<r_tran)[0]
    r3311 = r[idx3311]
    data3311 = data[:,idx3311]
    errs_3311 = np.sqrt(errs1[:,idx3311]**2 + errs2[:,idx3311]**2)
    mu_3311 = mu[idx3311]
    ##########################################################################
    # Set figure parameters
    gs = gridspec.GridSpec(4,5)
    gs.update(left=0.08, right=0.98, bottom = 0.06, top=0.98, hspace = 0.14,
               wspace=0.085)
    fig = plt.figure(1, figsize = (12, 9))
    xcb = 0.21
    ycb = [0.125, 0.435, 0.742]
    ##########################################################################
    nd = [3,2,2]
    color = r
    lgray = "0.85"
    dgray = "0.7"
    norm = Normalize(color.min(),color.max())
    ylims = [[9.7,10.2], [-2.4,1.2], [-0.5,0.75], [-3.5, 1.5]]
    tex, tex2, grad_tab1, grad_tab2 = [], [], [], []
    # Setting the colormap properties for the scatter plots
    cmap = brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colormap
    cmap = nc.cmap_discretize(cmap, 3)
    norm = Normalize(vmin=0, vmax=30)
    for i, (y, y1, y2) in enumerate(zip(data, errs1, errs2)):
        ######################################################################
        # Combined array ours + Loubser+ 2012
        rc = np.hstack((r, r_l12))
        rc1 = rc[rc <= r_tran]
        rc2 = rc[rc > r_tran]
        sbc = np.hstack((mu, l12_sb))
        sbc1 = sbc[rc <= r_tran]
        sbc2 = sbc[rc > r_tran]
        yc = np.hstack((y, l12[i]))
        ycerr1 = np.hstack((y1, l12_errs1[i]))
        ycerr2 = np.hstack((y2, l12_errs2[i]))
        ycerr = np.maximum(ycerr1, ycerr2)
        ######################################################################
        # Grouping according to radius includind data from Loubser
        # This is used for gradients and histograms
        ######################################################################
        yc1 = yc[rc <= r_tran]
        yc2 = yc[rc > r_tran]
        yc1err = ycerr[rc <= r_tran]
        yc2err = ycerr[rc > r_tran]
        ######################################################################
        # Clip values according to the models
        # for z, zerr in [[yc1, yc1err], [yc2, yc2err]]:
        #     for k in range(len(z)):
        #         if z[k] < ylims[i][0] or z[k] > ylims[i][1]:
        #             z[k] = np.nan
        #             zerr[k] = np.nan
        ######################################################################
        # Read data for sky p/m 1%
        ys = ['age', "metal", "alpha", "iron"]
        xs = ["r", 'mu']
        f1 = os.path.join(tables_dir, "pm1pc_{0}_{1}.txt".format(
                                     xs[0], ys[i]))
        f2 = os.path.join(tables_dir, "pm1pc_{0}_{1}.txt".format(
                                     xs[1], ys[i]))
        f3 = os.path.join(tables_dir, "pm6pc_{0}_{1}.txt".format(
                                     xs[0], ys[i]))
        f4 = os.path.join(tables_dir, "pm6pc_{0}_{1}.txt".format(
                                     xs[1], ys[i]))
        sim1 = np.loadtxt(f1).T
        sim2 = np.loadtxt(f2).T
        sim3 = np.loadtxt(f3).T
        sim4 = np.loadtxt(f4).T
        rms1 = interp1d(sim1[0], sim1[1], kind="linear", bounds_error=0,
                        fill_value=0.)
        rms2 = interp1d(sim2[0], sim2[1], kind="linear", bounds_error=0,
                        fill_value=0.)
        rms3 = interp1d(sim3[0], sim3[1], kind="linear", bounds_error=0,
                        fill_value=0.)
        rms4 = interp1d(sim4[0], sim4[1], kind="linear", bounds_error=0,
                        fill_value=0.)
        ######################################################################
        # Plot data in radial distance
        ######################################################################
        ax = plt.subplot(gs[i,0:2], xscale="linear")
        ax.minorticks_on()
        if i == 3:
            plt.xlabel(r"$\log \mbox{R} / \mbox{R}_e$")
        plt.ylabel(pars[i])

        ax2 = plt.subplot(gs[i,2:4], xscale="linear")
        ax2.minorticks_on()
        ii1 = np.where(sn >= 20.)[0]
        ii2 =  np.where(sn < 20.)[0]
        ax.errorbar(r, y, yerr = [y1, y2], fmt=None,
                    color=lgray, ecolor=lgray, capsize=0., zorder=1)
        ax.scatter(r, y, c=sn, s=50, cmap=cmap, zorder=2,
                   lw=0.5, norm=norm)
        ######################################################################
        # Plot data for Loubser 2012
        ax.errorbar(r_l12, l12[i], yerr = [l12_errs1[i], l12_errs2[i]],
                    color="r", ecolor=lgray,
                        fmt="s", mec="k", capsize=0,
                        alpha=1, ms=7.5, mew=0.5)
        # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlim(-1.5, 0.8)
        ######################################################################
        # Plot data for Coccato+ 2011
        ax.plot(rlodo, data_lodo[i], "^", mec="k", c="orange", ms=8, mew=0.5)
        #####################################################################
        # Draw arrows to indicate central limits
        ax.annotate("", xy=(-1., loubser[i]), xycoords='data',
        xytext=(-1.25, loubser[i]), textcoords='data',
        arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", ec="r",
                        lw=1.5))
        #####################################################################
        # Second plot: data as function of surface brightness
        #####################################################################
        ax2.errorbar(mu, y, yerr = [y1, y2], fmt=None,
                    color=lgray, ecolor=lgray, capsize=0., zorder=1)
        ax2.scatter(mu, y, c=sn, s=50, cmap=cmap, zorder=2,
                   lw=0.5, norm=norm)
        ax2.set_xlim(19,25)
        ######################################################################
        # Plot data for Loubser 2012
        ax2.errorbar(l12_sb, l12[i], yerr = [l12_errs1[i], l12_errs2[i]],
                    color="r", ecolor=lgray,
                        fmt="s", mec="k", capsize=0,
                        alpha=1, ms=7.5, mew=0.5)
        ######################################################################
        ######################################################################
        # Plot data for Coccato+ 2011
        ax2.plot(mulodo, data_lodo[i], "^", mec="k", c="orange", ms=8,
                 mew=0.5)
        #####################################################################
        if i == 3:
            plt.xlabel(r"$\mu_V$ (mag arcsec$^{-2}$)")
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
        ylim = plt.ylim()
        plt.minorticks_on()
        if i in [0,1,2]:
            ax.xaxis.set_ticklabels([])
            ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ax.set_ylim(ylims[i])
        ax2.set_ylim(ylims[i])
        ##################################################################
        # Draw arrows to indicate central limits
        ax2.annotate("", xy=(20.3, loubser[i]), xycoords='data',
        xytext=(19.55, loubser[i]), textcoords='data',
        arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", ec="r",
                        lw=1.5), zorder=100)
        ######################################################################
        # Measuring gradients
        ######################################################################
        # Inner halo
        ######################################################################
        popt1, pcov1 = curve_fit(f, rc1, yc1, sigma=yc1err)
        pcov1 = np.sqrt(np.diagonal(pcov1) + 0.01**2)
        label = r"{1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt1[1],2), round(pcov1[1],2),
                    round(popt1[0],2), round(pcov1[0],2))
        x = np.linspace(rc1.min(), r_tran, 100)
        yy = f(x, popt1[0], popt1[1])
        lll, = ax.plot(x, yy, "-k", lw=1.5, alpha=0.9, zorder=1000,
                       label=label)
        lll.set_dashes([10, 3, 2, 3])
        ax.annotate(r"$\Delta${0} (dex/dex)".format(
                    pars2[i], round(popt1[1],2), round(pcov1[1],2),
                    round(popt1[0],2), round(pcov1[0],2)),
                     xy=(0.06,0.35), xycoords="axes fraction", color="k",
                     fontsize=12)
        # ax.plot(x, yy + rms1(x), "-y")
        # ax.plot(x, yy - rms1(x), "-y")
        # ax.plot(x, yy + rms3(x), "-y")
        # ax.plot(x, yy - rms3(x), "-y")
        ax.fill_between(x, yy - rms1(x), yy + rms1(x), edgecolor="none",
                        color="0.5", linewidth=0, alpha=0.3)
        ######################################################################
        # Outer halo
        mask = ~np.isnan(yc2)
        if i == 0:
            popt2, pcov2 = curve_fit(f, rc2[mask], yc2[mask], sigma=yc2err)
        else:
            popt2, pcov2 = curve_fit(f, rc2[mask], yc2[mask])
        pcov2 = np.sqrt(np.diagonal(pcov2) + 0.01**2)
        label = r"{1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt2[1],2), round(pcov2[1],2),
                    round(popt2[0],2), round(pcov2[0],2))
        x = np.linspace(r_tran, rc2.max(),  100)
        yy = f(x, popt2[0], popt2[1])
        lll, = ax.plot(x, yy, "-k", lw=1.5, alpha=0.9, zorder=1000,
                       label=label)
        lll.set_dashes([10, 3])
        # ax.plot(x, yy + rms1(x), "-y")
        # ax.plot(x, yy - rms1(x), "-y")
        # ax.plot(x, yy + rms3(x), "-y")
        # ax.plot(x, yy - rms3(x), "-y")
        ax.fill_between(x, yy - rms1(x), yy + rms1(x), edgecolor="none",
                        color="0.5", linewidth=0, alpha=0.3)
        #####################################################################
        ax.legend(loc=3, fontsize=12, frameon=0, handlelength=2.5)
        #######################################################################
        ######################################################################
        # Measuring gradients - part II
        ######################################################################
        # Inner halo
        ######################################################################
        popt3, pcov3 = curve_fit(f, sbc1, yc1, sigma=yc1err)
        pcov3 = np.sqrt(np.diagonal(pcov3) + 0.01**2)
        label = r"{1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt3[1],2), round(pcov3[1],2),
                    round(popt3[0],2), round(pcov3[0],2))
        x = np.linspace(sbc1.min(), mu_tran, 100)
        yy = f(x, popt3[0], popt3[1])
        lll, = ax2.plot(x, yy, "-k", lw=1.5, alpha=0.9, zorder=1000,
                       label=label)
        lll.set_dashes([10, 3, 2, 3])
        ax2.annotate(r"$\Delta${0} (dex/(mag '' ''))".format(
                    pars2[i], round(popt1[1],2), round(pcov1[1],2),
                    round(popt1[0],2), round(pcov1[0],2)),
                     xy=(0.06,0.35), xycoords="axes fraction", color="k",
                     fontsize=12)
        # ax2.plot(x, yy + rms2(x), "-y")
        # ax2.plot(x, yy - rms2(x), "-y")
        # ax2.plot(x, yy + rms4(x), "-y")
        # ax2.plot(x, yy - rms4(x), "-y")
        ax2.fill_between(x, yy - rms2(x), yy + rms2(x), edgecolor="none",
                        color="0.5", linewidth=0, alpha=0.3)
        ######################################################################
        # Outer halo
        if i == 0:
            popt4, pcov4 = curve_fit(f, sbc2[mask], yc2[mask], sigma=yc2err[mask])
        else:
            popt4, pcov4 = curve_fit(f, sbc2[mask], yc2[mask])
        pcov4 = np.sqrt(np.diagonal(pcov4) + 0.01**2)
        label = r"{1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt4[1],2), round(pcov4[1],2),
                    round(popt4[0],2), round(pcov4[0],2))
        x = np.linspace(mu_tran, 24.5,  100)
        yy = f(x, popt4[0], popt4[1])
        lll, = ax2.plot(x, yy, "-k", lw=1.5, alpha=0.9, zorder=1000,
                       label=label)
        lll.set_dashes([10, 3])
        # ax2.plot(x, yy + rms2(x), "-y")
        # ax2.plot(x, yy - rms2(x), "-y")
        # ax2.plot(x, yy + rms4(x), "-y")
        # ax2.plot(x, yy - rms4(x), "-y")
        ax2.fill_between(x, yy - rms2(x), yy + rms2(x), edgecolor="none",
                        color="0.5", linewidth=0, alpha=0.3)
        #####################################################################
        ax2.legend(loc=3, fontsize=12, frameon=0, handlelength=2.5)
        #######################################################################
        # Model of Pipino and Matteucci 2005
        # if i == 1:
        #     xm = [-1., -0.3, 0.02]
        #     ym = [0.12, -0.03, -.14]
        #     ax.plot(xm, ym, "-b")
        #     print np.diff(ym) / np.diff(xm)
        # if i == 2:
        #     xm = [-1., -0.3, 0.02]
        #     ym = [0.15, 0.4, 0.5]
        #     ax.plot(xm, ym, "-b")
        #     print np.diff(ym) / np.diff(xm)
        ######################################################################
        ######################################################################
        # Create table line in latex
        tex.append(r"{0} & ${1[0]:.2f}\pm{2[0]:.2f}$ & ${1[1]:.2f}\pm{2[1]:.2f}$" \
            r" & ${3[0]:.2f}\pm{4[0]:.2f}$ & ${3[1]:.2f}\pm{4[1]:.2f}$""\\\\".format(
                pars2[i], popt1, pcov1, popt2, pcov2))
        tex2.append(r"{0} & ${1[0]:.2f}\pm{2[0]:.2f}$ & ${1[1]:.2f}\pm{2[1]:.2f}$" \
            r" & ${3[0]:.2f}\pm{4[0]:.2f}$ & ${3[1]:.2f}\pm{4[1]:.2f}$""\\\\".format(
                pars2[i], popt3, pcov3, popt4, pcov4))
        ######################################################################
        # Create output table
        grad_tab1.append(r"{0:15} | {1[0]:10.3f} | {2[0]:10.3f} | {1[1]:10.3f} | "
                         "{2[1]:10.3f} | {3[0]:10.3f} | {4[0]:10.3f} | {3[1]:10.3f} | "
                         "{4[1]:10.3f}".format(pars2[i], popt2, pcov2,
                                                     popt1, pcov1))
        grad_tab2.append(r"{0:15} | {1[0]:10.3f} | {2[0]:10.3f} | {1[1]:10.3f} | "
                         "{2[1]:10.3f} | {3[0]:10.3f} | {4[0]:10.3f} | {3[1]:10.3f} | "
                         "{4[1]:10.3f}".format(pars2[i], popt4, pcov4, popt3,
                                               pcov3))
        ######################################################################
        # Histograms
        ax3 = plt.subplot(gs[i, 4])
        ax3.minorticks_on()
        n, bins, patches = ax3.hist(yc, bins=20, histtype="step",
                                    orientation="horizontal", color=lgray,
                                    edgecolor="k", visible=0,
                                    range=ylims[i])
        ax3.hist(yc2, bins=bins, histtype="stepfilled",
                 orientation="horizontal", color=lgray,
                 edgecolor="none", alpha=0.8, visible=1,
                 label="R$>$R$_{\mbox{e}}$$")
        ax3.hist(yc1, bins=bins, histtype="stepfilled",
                 orientation="horizontal", color="b",
                 edgecolor="none", alpha=0.5, visible=1,
                 label="R$\leq$ R$_{\mbox{e}}$")
        leg = ax3.legend(loc=4, fontsize=12)
        leg.draw_frame(False)
        ax3.yaxis.set_ticklabels([])
        ax3.set_ylim(ylims[i])
        if i == 2:
            plt.xlabel(r"Frequency")
    for l in tex:
        print l + "\n"
    print "\n\n"
    for l in tex2:
        print l + "\n"
    plt.pause(0.0001)
    plt.savefig("figs/ssps_radius.png", dpi=100)
    # ##########################################################################
    # # Write gradients to tables
    # with open("gradients_logr.txt", "w") as f:
    #     f.write("# Gradients for the inner and outer halo\n")
    #     f.write("# Break radius at r={0:.2} r_e\n".format(r_tran))
    #     f.write("{0:15} | {1:10} | {2:10} | {3:10} | {2:10} | {4:10} | {2:10} "
    #             "| {5:10} | {2:10}\n".format("# Parameter", "Inn offset",
    #                                          "Error", "Inn Grad",
    #                                          "Out offset", "Out Grad"))
    #     f.write("\n".join(grad_tab1))
    # with open("gradients_sb.txt", "w") as f:
    #     f.write("# Gradients for the inner and outer halo\n")
    #     f.write("# Break SB at mu_v={0:.2} mag arcsec-2\n".format(mu_tran))
    #     f.write("{0:15} | {1:10} | {2:10} | {3:10} | {2:10} | {4:10} | {2:10} "
    #             "| {5:10} | {2:10}\n".format("# Parameter", "Inn offset",
    #                                          "Error", "Inn Grad",
    #                                          "Out offset", "Out Grad"))
    #     f.write("\n".join(grad_tab2))
    # ##########################################################################
    plt.show(block=True)
