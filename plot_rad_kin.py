# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:52:42 2013

@author: cbarbosa

Make plot of the kinematic properties as function of the position angle in 
radial bins
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import *
import canvas as cv

def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)
    return r, np.rad2deg(theta)

def polar2cart(r, theta):
    x = r * np.sin(np.deg2rad(theta))
    y = r * np.cos(np.deg2rad(theta))
    return x, y

def get_richtler():
    """ Retrieve data from Richtler et al. 2011. """
    files = [os.path.join(tables_dir, "results_n3311maj_sn30.txt"), 
             os.path.join(tables_dir, "results_n3311min_sn30.txt")]
    for i, fname in enumerate(files):
        r, v, verr, sig, sigerr = np.loadtxt(fname, usecols=(1,6,8,7,9)).T
        vmean = np.mean(v[r<1])
#        v -= vmean
        v -= 3873.
        if i == 0:
            phi = np.where(r > 0, 40 * np.ones_like(r), 
                           (40) * np.ones_like(r) - 180.)
        else:
            phi = np.where(r > 0, 115 * np.ones_like(r), 
                            (115.) * np.ones_like(r) - 180.) 
        x, y = polar2cart(np.abs(r), phi)
        ra = canvas.ra0 + x/3600.
        dec = canvas.dec0 + y/3600.
        x = canvas.arcsec2kpc(3600 * (ra - canvas.ra0))
        y = canvas.arcsec2kpc(3600 * (dec - canvas.dec0))         
        r, phi = cart2polar(x, y)
        table = np.column_stack((r, phi, v, verr, sig, sigerr))
        if i == 0:
            results = table
        else:
            results = np.vstack((results, table))
    nans = np.nan * np.ones_like(results[:,0])
    return np.column_stack((results, nans, nans, nans, nans, nans, nans))

def get_ventimiglia():
    """ Retrieve data from Ventimiglia et al. 2010. """
    ra, dec, v, verr, sig, sigerr= np.loadtxt(os.path.join(tables_dir, 
                                "ventimiglia_kin.txt"), usecols=np.arange(6)).T
    ra0v, dec0v = 159.17794743, -27.52809205 
    x = canvas.arcsec2kpc(3600 * (ra - ra0v))
    y = canvas.arcsec2kpc(3600 * (dec - dec0v))
    r, phi = cart2polar(x, y)
    vmean = np.mean(v[r<20])
    v -= vmean
    nans = np.nan * np.zeros_like(v)
    return np.column_stack((r, phi, v, verr, sig, sigerr, nans, nans, nans,
                            nans, nans, nans))

def get_our():
    """Retrieve our data points. """
    fname = os.path.join(work_dir, "results.tab")
    data = np.loadtxt(fname, usecols=(3,4,5,6,7,8,9,10,11,12))
    sn = np.loadtxt(fname, usecols=(14,))
    v = data[:,2]
    r = data[:,0]
    vmean = np.mean(v[r<5])
    data[:,2] -= vmean
    sndisp = sn / data[:,4] * 100
    sndisp_err = np.zeros_like(sndisp)
    return np.column_stack((data, sndisp, sndisp_err)), vmean, sn

def update_pa(tab, pa0):
    """ Offset the position angle of arrays by pa0. """
    pas = tab[:,1]
    pas -= pa0
    negpa = np.where(pas < -180.)
    pas[negpa] = pas[negpa] + 360.
    tab[:,1] = pas
    return tab
    
if __name__ == "__main__":
    work_dir = os.path.join(home, "single2")
    os.chdir(work_dir)
    # Definition of the bins to be ploted ####################################
    nslices = 8
    cone_ap = 360. / nslices # Cone aperture angle
    pas = np.arange(nslices/2) * cone_ap
    pas = np.column_stack((pas, pas+cone_ap, pas-180., pas-180. + cone_ap))
    pas = np.vstack((pas, [0, 180, -180, 0]))
    ##########################################################################
    # Definition of reference position angle
    pa_ref = pa0 - cone_ap/2
    ##########################################################################
    canvas = cv.CanvasImage("vband")
    plt.close()
    tab1 = get_richtler()
    tab2 = get_ventimiglia()
    tab3, v0, sn = get_our()
    # Offset PA by the value of pa0###
    tab1 = update_pa(tab1, pa_ref)
    tab2 = update_pa(tab2, pa_ref)
    tab3 = update_pa(tab3, pa_ref)
    # Central data to be included in all plots
    tab4 = tab3[tab3[:,0] < re]
    # tab3 = tab3[tab3[:,0] > re]
    ##################################
    # # Read results from the smoothed table
    # tabs = np.loadtxt("results_loess.dat", usecols=(1,2))
    # tab4 = np.copy(tab3[sn>sn_cut])
    # tab4[:,2] = tabs[:,0] - v0
    # tab4[:,4] = tabs[:,1]
    # tab4[:,3] = 0.
    # tab4[:,5] = 0.
    # #################################
    gs = gridspec.GridSpec(len(pas), 1)
    gs.update(left=0.14, right=0.98, bottom = 0.07, top=0.93, hspace = 0.12,
               wspace=0.085)
    ylims = [[-450, 450], [0, 850], [-.3, .4], [-.3, .4], [0, 8]]
    ylabels = [r"$V_{\rm{LOS}}$ (km/s)", r"$\sigma_{\rm{LOS}}$ (km/s)",
               r"$h_3$", r"$h_4$", r"S/N / $\sigma \times 100$"]
    fignames = ["figs/r_vel.png", "figs/r_sig.png", "figs/r_h3.png",
                "figs/r_h4.png", "figs/r_snsig.png"]
    maxloc = [7, 5, 5, 5, 5]
    alpha=1
    for ii in np.arange(5):
        fig = plt.figure(ii + 1, figsize=(6,8))
        for i, (p1, p2, p3, p4) in enumerate(pas):
            if ii == 0:
                print p1, p2, p3, p4
            ax = plt.subplot(gs[i, 0])
            ax.yaxis.grid(color="0.8", linestyle='dashed', alpha=0.5)
            plt.minorticks_on()
            labels = ["Richtler+ 2011", "Ventimiglia+ 2010", 
                          "This work", None]
            symbols = ["d", "^", "o", "o"]
            colors = ["g", "b", "r", "r"]
            mss = [5, 5, 5, 5]
            for j, t in enumerate([tab1, tab2, tab3]):
                indices = np.logical_and(t[:,1] >= p1, t[:,1] < p2)
                newt = t[indices]
                if len(newt) > 0:
                    newt = newt[newt[:,1].argsort()]
                    ax.errorbar(newt[:,0] , newt[:,2 + 2 * ii],
                                yerr = newt[:,3 + 2 * ii],
                                ecolor="0.5", capsize=0, c=colors[j],
                                fmt=symbols[j],
                                ms=mss[j], mec=colors[j], alpha=alpha)
                indices2 = np.logical_and(t[:,1] >= p3, t[:,1] < p4)
                newt2 = t[indices2]
                if len(newt2) > 0:
                    ax.errorbar(-newt2[:,0], newt2[:,2 + 2 * ii],
                                yerr = newt2[:,3 + 2 * ii],
                                ecolor="0.5", capsize=0, c=colors[j],
                                fmt=symbols[j],
                                ms=mss[j], mec=colors[j], alpha=alpha)
                idx = np.logical_and(tab4[:,1] >= pa0-90, tab4[:,1] <= pa0+90)
                newt = tab4[idx]
                ax.errorbar(newt[:,0], newt[:,2 + 2 * ii],
                            yerr = newt[:,3 + 2 * ii],
                            ecolor="0.5", capsize=0, c=colors[2],
                            fmt=symbols[2],
                            ms=mss[2], mec=colors[2], alpha=alpha)
                idx2 = np.bitwise_not(idx)
                newt = tab4[idx2]
                ax.errorbar(-newt[:,0], newt[:,2 + 2 * ii],
                            yerr = newt[:,3 + 2 * ii],
                            ecolor="0.5", capsize=0, c=colors[2],
                            fmt=symbols[2],
                            ms=mss[2], mec=colors[2], alpha=alpha)
            plt.ylim(ylims[ii][0], ylims[ii][1])
            plt.xlim(-45, 45)
            plt.ylabel(ylabels[ii])
            if ii in [0,2,3]:
                plt.axhline(ls="--", c="k")
            ax.annotate(r"$\phi = %d^{o}$,  $\Delta \phi = %d^{o}$" % (p1, p2-p1),
                        xy=(0.52,0.8), xycoords="axes fraction", color="k",
                        fontsize=12)
            if i + 1 < len(pas):
                ax.xaxis.set_ticklabels([])
            else:
                plt.xlabel(r"R [kpc]".format(int(pa0)))
            ax.axvline(x=0, c="k", ls="--")
            yloc = plt.MaxNLocator(maxloc[ii])
            ax.yaxis.set_major_locator(yloc)
            if ii==1:
                ax.axhline(647, c="y", ls="-", lw=1.5, alpha=0.5)
                ax.annotate(r"$\sigma_{{\mbox{{\small cluster}}}}$", xy=(0.12,0.8),
                            xycoords="axes fraction", color="y",
                            fontsize=12)
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax11 = ax.twiny()
            ax11.minorticks_on()
            ax11.plot(np.linspace(xlim[0] / re ,xlim[1] / re, 10),
                      np.linspace(ylim[0] ,ylim[1], 10),
                      "ok", visible=False)
            if i == 0:
                plt.xlabel("R / R$_e$")
            else:
                ax11.xaxis.set_ticklabels([])
            ax11.set_ylim(ylims[ii][0], ylims[ii][1])
        plt.savefig(fignames[ii])
        