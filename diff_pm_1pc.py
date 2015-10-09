#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 04/03/15 14:18

@author: Carlos Eduardo Barbosa

Calculating the differences in the stellar populations for 1% errors in sky
subtraction
"""
from diff_sim_lick import *

if __name__ == "__main__":
    cols = (0, 3, 82, 69, 72, 75, 84)
    data = mask_slits(os.path.join(home, "single2", "results.tab"),
                      cols=cols)
    datap = mask_slits(os.path.join(home, "p6pc", "results.tab"),
                      cols=cols)
    datam = mask_slits(os.path.join(home, "m6pc", "results.tab"),
                      cols=cols)
    ##########################################################################
    # Get intersection of data
    sref = list(set(data[0]) & set(datap[0]) & set(datam[0]))
    sref.sort()
    ##########################################################################
    # Filter and align data
    data = match_data(data[0].tolist(), sref, data[1:].T.astype(float)).T
    datam = match_data(datam[0].tolist(), sref, datam[1:].T.astype(float)).T
    datap = match_data(datap[0].tolist(), sref, datap[1:].T.astype(float)).T
    ##########################################################################
    data[2] += 9.
    datap[2] += 9.
    datam[2] += 9.
    ##########################################################################
    data[0] = np.log10(data[0]/re)
    #########################################################################
    # Some filtering
    # Removing low SB data
    ii = np.where(data[1] < 25)[0]
    data = data[:,ii]
    datap = datap[:,ii]
    datam = datam[:,ii]
    # Remove deviant ages
    ii = np.where(np.abs(data[2] - datap[2]) < 0.5)[0]
    data = data[:,ii]
    datap = datap[:,ii]
    datam = datam[:,ii]
    ##########################################################################
    fig = plt.figure(1, figsize=(8,8))
    gs = gridspec.GridSpec(4,2)
    gs.update(left=0.12, right=0.95, bottom = 0.1, top=0.95, hspace = 0.14,
               wspace=0.1)
    pars = [r"log Age (years)", r"[Z/H]", r"[$\alpha$/Fe]", r"[Fe/H]"]
    xlims = [[-.8, 0.8], [20,25]]
    rms_r, rms_sb = [], []
    ys = ['age', "metal", "alpha", "iron"]
    xs = ["r", 'mu']
    for i,j in enumerate((2,3,4,5)):
        ########################################
        # Differences as function of the radius
        ########################################
        idx = np.argwhere(np.isfinite(datap[j])).T
        idxm = np.argwhere(np.isfinite(datam[j])).T
        idxp =  np.argwhere(np.isfinite(data[j])).T
        idxm = np.intersect1d(idx, idxm)
        idxp = np.intersect1d(idx, idxp)
        diffp = datap[j][idxp] - data[j][idxp]
        diffm = datam[j][idxm] - data[j][idxm]
        diff = np.hstack((diffp, diffm))
        r = np.hstack((data[0][idxp], data[0][idxm]))
        mu = np.hstack((data[1][idxp], data[1][idxm]))
        for k,x in enumerate([r,mu]):
            idx_sorted = x.argsort()
            y = diff[idx_sorted]
            x = x[idx_sorted]
            xw = rolling_median(x, window=50, min_periods=1)
            yw = rolling_apply(y, 50, mad, min_periods=1)
            xw = np.hstack((xw, x.max()))
            yw = np.hstack((yw, yw[-1]))
            ax = plt.subplot(gs[i,k])
            ax.minorticks_on()
            ax.plot(data[k], diffp, "ob", mec="none", alpha=0.5,
                    label=r"+1\%")
            ax.plot(data[k], diffm, "or", mec="none", alpha=0.5,
                    label=r"-1\%")
            ax.axhline(y=0, ls="--", c="k")
            ax.fill_between(xw, yw, -yw, edgecolor="none", color="y",
                            linewidth=0, alpha=0.5)
            if k == 1:
                ax.yaxis.set_ticklabels([])
            if i != 3:
                ax.xaxis.set_ticklabels([])
            else:
                xlabels = [r"$\log R / R_e$", r"$\mu_V$ (mag arcsec$^{-2}$)"]
                plt.xlabel(xlabels[k])
            if k == 0:
                plt.ylabel(r"$\Delta${0}".format(pars[i]))
            if i == 0:
                plt.legend(loc=0, ncol=2, prop={'size':10})
            ax.set_xlim(xlims[k])
            filename = os.path.join(tables_dir, "pm6pc_{0}_{1}.txt".format(
                                     xs[k], ys[i]))
            np.savetxt(filename, np.column_stack((xw, yw)))
    plt.pause(0.001)
    plt.savefig(os.path.join(figures_dir, "sky_pm_6pc.png"))
    plt.show(block=True)


