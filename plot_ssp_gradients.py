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
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
import cap_loess_2d as ll
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from config import *
import newcolorbars as nc

def get_model_range(table):
    """ Get the range for the indices according to models. """
    modeldata = np.loadtxt(table)
    indices = modeldata[:,3:].T
    ranges = np.zeros((len(indices), 2))
    for i, index in enumerate(indices):
        ranges[i] = [index.min(), index.max()]
    return ranges

def f(x, zp, grad ):
    return zp + grad * x

def residue(p, x, y, yerr):
    return (p[1] * x +    p[0] - y) / yerr

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    #including valid will REQUIRE there to be enough datapoints.
    #for example, if you take out valid, it will start @ point one,
    #not having any prior points, so itll be 1+0+0 = 1 /3 = .3333
    smas = np.convolve(values, weigths, 'same')
    return smas # as a numpy array


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
    table = "results.tab"
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
    rl = np.array([42.8, 50.2, 57.1, 67.2, 108.4, 108.4]) / re / 4.125
    pal = np.array([47.1, 83.7, 10.3, 102.5, 118.8, 64.])
    rl = np.log10(rl)
    al = np.log10(np.array([13.5, 13.5, 13.5, 13.5, 13.5, 13.5]))
    ml = np.array([-.46, -.33, -.73, -.39, -.34, -.85])
    alpl = np.array([.45, .48, .5, .44, .5, -.03])
    ld = np.column_stack((al, ml, alpl)).T
    ##########################################################################
    # Central values according to Loubser+ 2009
    loubser = np.array([np.log10(np.power(10, 0.94) * 10**9),
                        0.12, 0.4, 0.12 -0.94 * 0.4 ])
    ##########################################################################
    # Stellar populations by Loubser + 2012
    l12_table = os.path.join(tables_dir, "loubser12_populations.txt")
    r_l12 = np.loadtxt(l12_table, usecols=(0,)) + np.log10(26.6/re)
    l12 = np.loadtxt(l12_table, usecols=(1,4,7,4)).T
    l12[0] *= 10**9
    l12[3] += -0.94 * l12[2]
    l12_errs1= np.loadtxt(l12_table, usecols=(2,5,8,5)).T
    l12_errs1[0] *= 10**9
    l12_errs1[3] += -0.94 * l12_errs1[2]
    l12_errs2 = np.loadtxt(l12_table, usecols=(3,6,9,6)).T
    l12_errs2[0] *= 10**9
    l12_errs2[3] += -0.94 * l12_errs2[2]
    l12_sb = np.loadtxt(l12_table, usecols=(10,))
    for i in range(3):
        l12_errs1[i] = l12[i] - l12_errs1[i]
        l12_errs2[i] = l12_errs2[i] - l12[i]
    l12_errs1[0] = l12_errs1[0] / l12[0] * np.log10(np.e)
    l12_errs2[0] = l12_errs2[0] / l12[0] * np.log10(np.e)
    l12[0] = np.log10(l12[0])
    diff = np.log10(15 * 10**9) - l12[0]
    l12_errs2[0] = np.minimum(diff, l12_errs2[0])
    ##########################################################################
    # Read SSP parameters and convert errors for plot
    data = np.loadtxt(table, usecols=(69, 72, 75, 84)).T
    data[0] *= 10**9
    x, y, sn = np.loadtxt(table, usecols=(1,2,14)).T
    errs1 = np.loadtxt(table, usecols=(70,73,76,85)).T
    errs2 = np.loadtxt(table, usecols=(71,74,77,86)).T
    errs1[0] *= 10**9
    errs2[0] *= 10**9
    for i in range(4):
        errs1[i] = data[i] - errs1[i]
        errs2[i] = errs2[i] - data[i]
    errs1[0] = errs1[0] / data[0] * np.log10(np.e)
    errs2[0] = errs2[0] / data[0] * np.log10(np.e)
    data[0] = np.log10(data[0])
    diff = np.log10(15*10**9) - data[0]
    errs2[0] = np.minimum(diff, errs2[0])
    ##########################################################################
    # Smooth data
    sn_thres = 25.
    frac_loess = 0.3
    data2 = np.copy(data)
    good = np.arange(len(data[0]))
    sn_high = np.where(((~np.isnan(sn)) & (sn>=sn_thres)))[0]
    sn_low = np.delete(good, sn_high)
    for i,vector in enumerate(data):
        vector_low = ll.loess_2d(x[sn_low], y[sn_low], vector[sn_low],
                                frac=frac_loess)
        for j,v in zip(sn_low, vector_low):
            data2[i,j] = v
    ##########################################################################
    # Bin data in r
    rbinnum, redges = np.histogram(r, bins=5, range=(r_tran, r.max()))
    data_r = []
    rbins = []
    errs_r = []
    for i, bin in enumerate(rbinnum):
        idx = np.logical_and(r > redges[i], r< redges[i+1])
        if not len(np.where(idx)[0]):
            continue
        if lum_weight:
            data_r.append(np.average(data[:,idx].T, axis=0,
                                     weights=np.power(10, -0.4*mu[idx])))
            rbins.append(np.average(r[idx], axis=0,
                                     weights=np.power(10, -0.4*mu[idx])))
        else:
            data_r.append(np.average(data[:,idx].T, axis=0,
                                     weights=np.ones_like(mu[idx])))
            rbins.append(np.average(r[idx], axis=0,
                                     weights=np.ones_like(mu[idx])))
        sigma = np.std(data[:,idx].T, axis=0)
        errorbars = np.median(errs1[:,idx] + errs2[:,idx], axis=1)
        errs_r.append(np.sqrt(sigma**2 + errorbars**2))
    data_r = np.array(data_r)
    rbins = np.array(rbins)
    errs_r = np.array(errs_r)
    ##########################################################################
    # Bin data in mu
    mubinnum, muedges = np.histogram(mu, bins=5, range=(mu_tran, 24.5))
    data_mu = []
    mubins = []
    errs_mu = []
    for i, bin in enumerate(mubinnum):
        idx = np.logical_and(mu > muedges[i], mu< muedges[i+1])
        if not len(np.where(idx)[0]):
            continue
        # print data[:,idx]
        median = False
        if median:
            d = data[:,idx].T
            med = np.median(d, axis=0)
            data_mu.append(med)
            mubins.append(np.median(mu[idx], axis=0))
            sigma =  np.median(np.abs(d - med), axis=0)
        else:
            if lum_weight:
                data_mu.append(np.average(data[:,idx].T, axis=0,
                                         weights=np.power(10, -0.4*mu[idx])))
                mubins.append(np.average(mu[idx], axis=0,
                                         weights=np.power(10, -0.4*mu[idx])))
            else:
                data_mu.append(np.average(data[:,idx].T, axis=0,
                                         weights=np.ones_like(mu[idx])))
                mubins.append(np.average(mu[idx], axis=0,
                                         weights=np.ones_like(mu[idx])))
            sigma = np.std(data[:,idx].T, axis=0)
        errorbars = np.median(errs1[:,idx] + errs2[:,idx], axis=1)
        errs_mu.append(np.sqrt(sigma**2 + errorbars**2))
    data_mu = np.array(data_mu)
    mubins = np.array(mubins)
    errs_mu = np.array(errs_mu)
    #########################################################################
    # Taking only inner region for gradients in NGC 3311
    idx3311 = np.where(r<r_tran)[0]
    r3311 = r[idx3311]
    data3311 = data[:,idx3311]
    errs_3311 = np.sqrt(errs1[:,idx3311]**2 + errs2[:,idx3311]**2)
    mu_3311 = mu[idx3311]
    ##########################################################################
    if restrict_pa:
        idx = np.logical_and(pa > 0, pa < 90)
        idx2 = np.bitwise_not(idx)
    else:
        idx = np.arange(len(pa)) # removing HCC 007 and NGC 3309
        idx2 = idx
    ##########################################################################
    # Filter data
    data = np.transpose(np.transpose(data)[idx])
    data2 = np.transpose(np.transpose(data2)[idx])
    errs1 = np.transpose(np.transpose(errs1)[idx])
    errs2 = np.transpose(np.transpose(errs2)[idx])
    r = r[idx]
    pa = pa[idx]
    sn = sn[idx]
    mu = mu[idx]
    ##########################################################################
    # Set figure parameters
    gs = gridspec.GridSpec(4,5)
    gs.update(left=0.075, right=0.98, bottom = 0.06, top=0.98, hspace = 0.12,
               wspace=0.085)
    fig = plt.figure(1, figsize = (10.5, 9))
    xcb = 0.21
    ycb = [0.125, 0.435, 0.742]
    ##########################################################################
    nd = [3,2,2]
    cmap = nc.cmap_discretize(cm.get_cmap("rainbow"), 4)
    cmap="gray"
    color = r
    lgray = "0.6"
    dgray = "0.1"
    norm = Normalize(color.min(),color.max())
    ylims = [[8.5,10.5], [-2.4,1.2], [-0.5,0.75], [-3, 1.5]]
    tex, tex2, grad_tab1, grad_tab2 = [], [], [], []
    for i, (y, y1, y2, z) in enumerate(zip(data, errs1, errs2, data2)):
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
        ycerr = np.sqrt(ycerr1**2 + ycerr2**2)
        # Grouping according to radius
        yc1 = yc[rc <= r_tran]
        yc2 = yc[rc > r_tran]
        yc1err = ycerr[rc <= r_tran]
        yc2err = ycerr[rc > r_tran]
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
        ax.errorbar(r[ii1], y[ii1], yerr = [y1[ii1], y2[ii1]], fmt="x",
                    color=dgray, ecolor=lgray, capsize=0, mec=dgray, ms=6,
                    alpha=0.5)
        ax.errorbar(r[ii2], y[ii2], yerr = [y1[ii2], y2[ii2]], fmt="x",
                    color=lgray, ecolor=lgray, capsize=0, mec=lgray, ms=6,
                    alpha=0.5)
        # ax.errorbar(r, y, yerr = [y1, y2], fmt="o",
        #             color=lgray, ecolor=lgray, capsize=0, mec=lgray, ms=6,
        #             alpha=1)
        ax.errorbar(rbins, data_r[:,i], yerr = errs_r[:,i], fmt="s",
                    color="r", ecolor="r", capsize=0, mec="r", ms=8,
                    zorder=100)
        ######################################################################
        # Plot data for Loubser 2012
        ax.errorbar(r_l12, l12[i], yerr = [l12_errs1[i], l12_errs2[i]],
                    fmt="*", color="g", ecolor="g",
                    capsize=0, mec="g", ms=8, alpha=0.7)
        ######################################################################
        # Plot moving average
        # Moving average
        # ind = np.argsort(rc)
        # ax.plot(rc[ind], movingaverage(yc[ind], 10), "-", color="0.2")
        # ind2 = np.argsort(sbc)
        # ax2.plot(sbc[ind2], movingaverage(yc[ind2], 10), "-", color="0.2")
        ######################################################################
        # sp = ax.scatter(r,y,c=pa, zorder=2, cmap=cmap, norm=norm, s=40)
        # ax.scatter(rl,ld[i], c=pal, zorder=2, cmap=cmap, norm=norm, s=40,
        #            marker="s")
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.set_xlim(-1.3, 0.7)

        ##################################################################
        # Draw arrows to indicate central limits
        ax.annotate("", xy=(-1., loubser[i]), xycoords='data',
        xytext=(-1.25, loubser[i]), textcoords='data',
        arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", ec="g",
                        lw=2))
        #####################################################################
        # Read data for sky p/m 1%
        rsky, sky = np.loadtxt(os.path.join(tables_dir, "rms_1pc_re.txt"),
                               usecols=(0,i+1)).T
        rms = interp1d(rsky, sky, kind="linear", bounds_error=0, fill_value=0.)
        musky, sky2 = np.loadtxt(os.path.join(tables_dir, "rms_1pc_sb.txt"),
                               usecols=(0,i+1)).T
        rms2 = interp1d(musky, sky2, kind="linear", bounds_error=0,
                        fill_value=0.)
        #####################################################################
        # Second plot: data as function of surface brightness
        #####################################################################
        ax2.errorbar(mu[ii1], y[ii1], yerr = [y1[ii1], y2[ii1]], fmt="x",
                    color=dgray, ecolor=lgray, capsize=0, mec=dgray, ms=6,
                    alpha=0.5)
        ax2.errorbar(mu[ii2], y[ii2], yerr = [y1[ii2], y2[ii2]], fmt="x",
                    color=lgray, ecolor=lgray, capsize=0, mec=lgray, ms=6,
                    alpha=0.5)
        # ax2.plot(mu,z,"x", ms=7, color=dgray)
        ax2.errorbar(mubins, data_mu[:,i], yerr = errs_mu[:,i], fmt="s",
                    color="r", ecolor="r", capsize=0, mec="r", ms=8,
                    zorder=100)
        ######################################################################
        # Plot data for Loubser 2012
        ax2.errorbar(l12_sb, l12[i], yerr = [l12_errs1[i], l12_errs2[i]],
                    fmt="*", color="g", ecolor="g",
                    capsize=0, mec="g", ms=8, alpha=0.7)
        ######################################################################
        if i == 3:
            plt.xlabel(r"$\mu_V$ (mag arcsec$^{-2}$)")
        ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
        ylim = plt.ylim()
        plt.minorticks_on()
        if i in [0,1,2]:
            ax.xaxis.set_ticklabels([])
            ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])
        ##################################################################
        # Draw arrows to indicate central limits
        ax2.annotate("", xy=(20.3, loubser[i]), xycoords='data',
        xytext=(19.55, loubser[i]), textcoords='data',
        arrowprops=dict(arrowstyle="<-", connectionstyle="arc3", ec="g",
                        lw=2), zorder=100)
        ######################################################################
        # Measuring gradients
        ######################################################################
        popt, pcov = curve_fit(f, rbins, data_r[:,i], sigma=errs_r[:,i])
        pcov = np.sqrt(np.diagonal(pcov) + 0.01**2)
        x = np.linspace(r_tran, rbins.max()+0.1, 100)

        yy = f(x, popt[0], popt[1])
        ax.plot(x, yy, "--r", lw=2)
        ax.fill_between(x, rms(x) + yy, yy -rms(x), edgecolor="none", color="y",
                        linewidth=0, alpha=0.4)
        ax.annotate(r"-- -- {1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt[1],2), round(pcov[1],2),
                    round(popt[0],2), round(pcov[0],2)),
                     xy=(0.06,0.12), xycoords="axes fraction", color="r",
                     fontsize=12)
        if i in [0,2]:
            # Calculating gradient only in the upper points
            idx_above = np.logical_and(y >= f(r,popt[0], popt[1]), r > r_tran)
            rab = r[idx_above]
            yab = y[idx_above]
            y1ab = y1[idx_above]
            y2ab = y2[idx_above]
            yaberr = np.maximum(y1ab, y2ab)
            poptab, pcovab = curve_fit(f, rab, yab, sigma=yaberr)
            pcovab = np.sqrt(np.diagonal(pcovab) + 0.01**2)
            x = np.linspace(r_tran, rab.max(), 100)
            yy = f(x, poptab[0], poptab[1])
            ax.plot(x, yy, "--b", lw=2)
            # Calculating gradient only in the bottom points
            idx_above = np.logical_and(y < f(r,popt[0], popt[1]), r > r_tran)
            rab = r[idx_above]
            yab = y[idx_above]
            y1ab = y1[idx_above]
            y2ab = y2[idx_above]
            yaberr = np.maximum(y1ab, y2ab)
            poptab, pcovab = curve_fit(f, rab, yab, sigma=yaberr)
            pcovab = np.sqrt(np.diagonal(pcovab) + 0.01**2)
            x = np.linspace(r_tran, rab.max(), 100)
            yy = f(x, poptab[0], poptab[1])
            ax.plot(x, yy, "--b", lw=2)
        ######################################################################
        popt2, pcov2 = curve_fit(f, rc1, yc1, sigma=yc1err)
        pcov2 = np.sqrt(np.diagonal(pcov2) + 0.01**2)
        x = np.linspace(rc1.min(), r_tran, 100)
        yy = f(x, popt2[0], popt2[1])
        ax.plot(x, yy, "--k", lw=2)
        ax.fill_between(x, rms(x) + yy, yy -rms(x), edgecolor="none", color="y",
                        linewidth=0, alpha=0.4)
        ax.annotate(r"$\Delta${0} (dex/dex)".format(
                    pars2[i], round(popt2[1],2), round(pcov2[1],2),
                    round(popt2[0],2), round(pcov2[0],2)),
                     xy=(0.06,0.32), xycoords="axes fraction", color="k",
                     fontsize=12)
        ax.annotate(r"-- -- {1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt2[1],2), round(pcov2[1],2),
                    round(popt2[0],2), round(pcov2[0],2)),
                     xy=(0.06,0.22), xycoords="axes fraction", color="k",
                     fontsize=12)
        ######################################################################
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
        popt3, pcov3 = curve_fit(f, mubins, data_mu[:,i], sigma=errs_mu[:,i])
        pcov3 = np.sqrt(np.diagonal(pcov3) + 0.01**2)
        x = np.linspace(mu_tran, 24.6, 100)
        yy = f(x, popt3[0], popt3[1])
        ax2.plot(x, yy, "--r", lw=2)
        ax2.fill_between(x, rms2(x) + yy, yy -rms2(x), edgecolor="none",
                        color="y", linewidth=0, alpha=0.4)
        ax2.annotate(r"-- -- {1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt3[1],2), round(pcov3[1],2),
                    round(popt3[0],2), round(pcov3[0],2)),
                     xy=(0.06,0.12), xycoords="axes fraction", color="r",
                     fontsize=12)
        if i in [0,2]:
            # Calculating gradient only in the upper points
            idx_above = np.logical_and(y >= f(mu,popt3[0], popt3[1]), mu > mu_tran)
            muab = mu[idx_above]
            yab = y[idx_above]
            y1ab = y1[idx_above]
            y2ab = y2[idx_above]
            yaberr = np.maximum(y1ab, y2ab)
            poptab, pcovab = curve_fit(f, muab, yab, sigma=yaberr)
            pcovab = np.sqrt(np.diagonal(pcovab) + 0.01**2)
            x = np.linspace(mu_tran, 24.6, 100)
            yy = f(x, poptab[0], poptab[1])
            ax2.plot(x, yy, "--b", lw=2)
            # Calculating gradient only in the bottom points
            # Calculating gradient only in the upper points
            idx_above = np.logical_and(y < f(mu,popt3[0], popt3[1]), mu > mu_tran)
            muab = mu[idx_above]
            yab = y[idx_above]
            y1ab = y1[idx_above]
            y2ab = y2[idx_above]
            yaberr = np.maximum(y1ab, y2ab)
            poptab, pcovab = curve_fit(f, muab, yab, sigma=yaberr)
            pcovab = np.sqrt(np.diagonal(pcovab) + 0.01**2)
            x = np.linspace(mu_tran, 24.6, 100)
            yy = f(x, poptab[0], poptab[1])
            ax2.plot(x, yy, "--b", lw=2)
        ######################################################################
        popt4, pcov4 = curve_fit(f, sbc1, yc1, sigma=yc1err)
        pcov4 = np.sqrt(np.diagonal(pcov4) + 0.01**2)
        x = np.linspace(sbc1.min(), mu_tran, 10)
        yy = f(x, popt4[0], popt4[1])
        ax2.plot(x, yy, "--k", lw=2)
        ax2.fill_between(x, rms2(x) + yy, yy -rms2(x), edgecolor="none",
                        color="y", linewidth=0, alpha=0.4)
        ax2.annotate(r"$\Delta$ {0} \small{{(dex mag$^{{-1}}$arcsec$^2$)}}".format(
                    pars2[i], round(popt4[1],2), round(pcov4[1],2),
                    round(popt4[0],2), round(pcov4[0],2)),
                     xy=(0.06,0.32), xycoords="axes fraction", color="k",
                     fontsize=12)
        ax2.annotate(r"-- -- {1:.2f}$\pm${2:.2f}".format(
                    pars2[i], round(popt4[1],2), round(pcov4[1],2),
                    round(popt4[0],2), round(pcov4[0],2)),
                     xy=(0.06,0.22), xycoords="axes fraction", color="k",
                     fontsize=12)
        #######################################################################
        # ax.set_xlim(0,1.6)
        ax2.set_xlim(19,25)
        ax.set_ylim(ylims[i])
        ax2.set_ylim(ylims[i])
        # ax.axvline(x=r_tran, c="k", ls="-.")
        # ax2.axvline(x=mu_tran, c="k", ls="-.")
        ######################################################################
        # Create table line in latex
        tex.append(r"{0} & ${1[0]:.2f}\pm{2[0]:.2f}$ & ${1[1]:.2f}\pm{2[1]:.2f}$" \
            r" & ${3[0]:.2f}\pm{4[0]:.2f}$ & ${3[1]:.2f}\pm{4[1]:.2f}$""\\\\".format(
                pars2[i], popt2, pcov2, popt, pcov))
        tex2.append(r"{0} & ${1[0]:.2f}\pm{2[0]:.2f}$ & ${1[1]:.2f}\pm{2[1]:.2f}$" \
            r" & ${3[0]:.2f}\pm{4[0]:.2f}$ & ${3[1]:.2f}\pm{4[1]:.2f}$""\\\\".format(
                pars2[i], popt4, pcov4, popt3, pcov3))
        ######################################################################
        # Create output table
        grad_tab1.append(r"{0:15} | {1[0]:10.3f} | {2[0]:10.3f} | {1[1]:10.3f} | "
                         "{2[1]:10.3f} | {3[0]:10.3f} | {4[0]:10.3f} | {3[1]:10.3f} | "
                         "{4[1]:10.3f}".format(pars2[i], popt2, pcov2,
                                                     popt, pcov))
        grad_tab2.append(r"{0:15} | {1[0]:10.3f} | {2[0]:10.3f} | {1[1]:10.3f} | "
                         "{2[1]:10.3f} | {3[0]:10.3f} | {4[0]:10.3f} | {3[1]:10.3f} | "
                         "{4[1]:10.3f}".format(pars2[i], popt4, pcov4, popt3,
                                               pcov3))
        ######################################################################
        # Histograms
        ax3 = plt.subplot(gs[i, 4])
        ax3.minorticks_on()
        n, bins, patches = ax3.hist(yc, bins=15, histtype="step",
                                    orientation="horizontal", color=lgray,
                                    edgecolor="k", visible=0)
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
        # ax3.axvline(x=0, ls="--", c="k")
        # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        # ax3.xaxis.set_ticklabels([])
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
    ##########################################################################
    # Write gradients to tables
    with open("gradients_logr.txt", "w") as f:
        f.write("# Gradients for the inner and outer halo\n")
        f.write("# Break radius at r={0:.2} r_e\n".format(r_tran))
        f.write("{0:15} | {1:10} | {2:10} | {3:10} | {2:10} | {4:10} | {2:10} "
                "| {5:10} | {2:10}\n".format("# Parameter", "Inn offset",
                                             "Error", "Inn Grad",
                                             "Out offset", "Out Grad"))
        f.write("\n".join(grad_tab1))
    with open("gradients_sb.txt", "w") as f:
        f.write("# Gradients for the inner and outer halo\n")
        f.write("# Break SB at mu_v={0:.2} mag arcsec-2\n".format(mu_tran))
        f.write("{0:15} | {1:10} | {2:10} | {3:10} | {2:10} | {4:10} | {2:10} "
                "| {5:10} | {2:10}\n".format("# Parameter", "Inn offset",
                                             "Error", "Inn Grad",
                                             "Out offset", "Out Grad"))
        f.write("\n".join(grad_tab2))
    ##########################################################################
    plt.show(block=True)
