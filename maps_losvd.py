# -*- coding: utf-8 -*-
"""
Created on Mon March 9th 2014

Produces maps for LOSVD for the Hydra I cluster core

@author: cbarbosa
"""
from __future__ import division
import os

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

from config import *
import canvas as cv
import cap_loess_2d as ll
from voronoi_polygons import voronoi_polygons
import newcolorbars as nc

def set_canvas(plot_residual):
    """ Set canvas according to type of contours. """
    if plot_residual:
        canvas = cv.CanvasImage("residual")
        canvas.data = np.clip(canvas.data, 1., canvas.data.max())
        canvas.data = -2.5 * np.log10(
                          canvas.data/480./canvas.ps**2) +  27.2
        yc, xc, r = 775, 1251, 90
        for x in np.arange(xc-r, xc+r):
            for y in np.arange(yc-r, yc+r):
                if (x - xc)**2 + (y - yc)**2 < r**2:
                    canvas.data[x,y] = np.nan
    else:
        canvas = cv.CanvasImage("vband")
        canvas.data = np.clip(canvas.data - 4900., 1., canvas.data)
        canvas.data = -2.5 * np.log10(
                      canvas.data/480./canvas.ps/canvas.ps) +  27.2

    return canvas

def make_voronoi(xy, xy2=None, rout=40.):
    """ Produce Voronoi tesselation of a set of positions. 
    
    ================
    Input parameters 
    ================  
    xy : array
        Reference points for tesselation
    xy2 : array
        Additional points to include holes 
    rout : float
        Maximum radius to extend tesselation.
    
    """ 
    nbins = len(xy)
    if xy2 is not None:
        xy = np.concatenate((xy, xy2))
    circle = cv.circle_xy(40.)
    points = np.concatenate((xy, circle))
    polygons = np.array(voronoi_polygons(points))[:nbins]
    return polygons

def get_positions(specs):
    """ Matches two different tables using the spectra column. """
    xy = []
    for i, spec in enumerate(specs):
        slit = spec.split("n3311", 1)[1].replace(".fits", "")
        # slit = spec.split(".")[0].split("_", 1)[1][5:]
        index = canvas.slits.ids.index(slit)
        xy.append([canvas.slits.x[index], canvas.slits.y[index]])
    return np.array(xy)

def get_positions_by_slits(slits):
    """ Matches two different tables using the spectra column. """
    xy = []
    for i, slit in enumerate(slits):
        index = canvas.slits.ids.index(slit)
        xy.append([canvas.slits.x[index], canvas.slits.y[index]])
    return np.array(xy)

    
def get_coords(specs):
    slits = cv.Slitlets()
    coords = []
    for spec in specs:
        region = spec.replace(".fits", "").split("_", 1)[1][5:]
        index = slits.ids.index(region)
        coords.append([slits.ra[index],  slits.dec[index]])
    return np.array(coords)

def merge_tables():
    """ Merge all tables into a single file to be used consistently for all
        the maps.
        The version in this file is abridged to include only kinematic data.
        """
    filename = "ppxf_all.dat"
#    files = ["ppxf_results2.dat", "lick2.tsv", "ages_Z_alpha2.tsv",
#             "lick_errs2.tsv"]
    s1 = np.genfromtxt(filename, usecols=(0,), dtype=None).tolist()
    sref = s1[:]
    sref.sort()
    x,y = get_positions(sref).T
    r = np.sqrt(x*x + y*y)
    pa = np.rad2deg(np.arctan2(x, y))
    data1 = np.loadtxt(filename, usecols=np.arange(1,11))
    ##########################################################################
    # Account for difference in resolution
    # Not used anymore because the resolution is now matched in pPXF
    # fwhm_dif = (2.5 - 2.1) * c / 5500. / 2.3548
    # data1[:,2] = np.sqrt(data1[:,2]**2 - fwhm_dif**2)
    ##########################################################################
    data1 = match_data(s1, sref, data1)
    results = np.column_stack((sref, x, y, r, pa, data1))
    header = ['FILE', "X[kpc]", "Y[kpc]",
              "R[kpc]", "PA",
              'V', 'dV', 'S', 'dS', 'h3', 'dh3',
              'h4', 'dh4',  'chi/DOF', 'S/N']
    with open(outtable, "w") as f:
        for i,field in enumerate(header):
            print "# {0} : {1}\n".format(i, field)
            f.write("# {0} : {1}\n".format(i, field))
        np.savetxt(f, results, fmt="%s")
    return

def match_data(s1, s2, d1):
    idx = np.array([s1.index(x) for x in s2])
    return d1[idx]

def polar2cart(r, theta):
    x = r * np.sin(np.deg2rad(theta))
    y = r * np.cos(np.deg2rad(theta))
    return x, y

def get_richtler():
    """ Retrieve data from Richtler et al. 2011. """
    files = [os.path.join(tables_dir, "results_n3311maj_sn30.txt"),
             os.path.join(tables_dir, "results_n3311min_sn30.txt")]
    tables = []
    for i, fname in enumerate(files):
        r, v, verr, sig, sigerr = np.loadtxt(fname, usecols=(1,6,8,7,9)).T
        vmean = np.mean(v[r<1])
#        v -= vmean
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
        table = np.column_stack((x, y, v, sig))
        tables.append(table)
    return tables

def get_ventimiglia():
    """ Retrieve data from Ventimiglia et al. 2010. """
    ra, dec, v, verr, sig, sigerr= np.loadtxt(os.path.join(tables_dir,
                                "ventimiglia_kin.txt"), usecols=np.arange(6)).T
    ra0v, dec0v = 159.17794743, -27.52809205
    x = 1.25 * canvas.arcsec2kpc(3600 * (ra - canvas.ra0))
    y = 1.25 * canvas.arcsec2kpc(3600 * (dec - canvas.dec0))
    vmean = np.mean(v[x**2+y**2<20])
    return np.column_stack((x, y, v, sig))
    
def make_sn():
    """ Produces a map of the signal to noise per bin according to pPXF. """
    ###############################################
    # Read values of S/N
    sn = 1.8 * np.loadtxt(outtable, usecols=(14,))
    ###############################################
    # Find good (and bad) regions according to S/N
    good = np.where(((~np.isnan(sn)) & (sn>=sn_cut)))[0]
    ###############################################
    # Filter S/N
    sn = sn[good]
    ###############################################
    # Colorbar limits
    vmin, vmax = 10, 50
    # Set limits for the plot
    norm = Normalize(vmin, vmax)
    cmap = "Spectral"
    # Produces a collection of polygons with colors according to S/N values
    coll = PolyCollection(polygons_bins[good], array=sn, cmap=cmap,
                          edgecolors='w', norm=norm, linewidths=1.)
    ###############################################
    # Initiate figure and axis for matplotlib
    fig, ax = plt.subplots(1, 1, figsize=(6.4,6), )
    fig.subplots_adjust(left=0.09, right=0.985, bottom = 0.092, top=0.98,
                        hspace = 0.05, wspace=0.06)
    ###############################################
    # ax.add_patch(Rectangle((-100, -100), 200, 200, facecolor="0.8", zorder=0,
    #                        alpha=0.5))
    ###############################################
    # Draw the polygons
    draw_map(fig, ax, coll, lims=40)
    ###############################################
    # Add contours according to V-band image
    # draw_contours("residual", fig, ax, c="k")
    draw_contours("vband", fig, ax, c="k")
    # Draw actual slit positions
    canvas.draw_slits(ax, slit_type=1, fc="r", ec="r", ignore=ignore_slits )
    canvas.draw_slits(ax, slit_type=3, fc="r", ec="r", ignore=ignore_slits )
    ###############################################
    # Draw white rectangle in the position of the colorbar so background
    # stars do not overplot the labels and ticks
    plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10000,
                        color="w"))
    ###############################################
    # Draw the colorbar
    draw_colorbar(fig, ax, coll, ticks=np.linspace(vmin,vmax,5),
                  cblabel=r"S/N [\AA$^{-1}$]", cbar_pos=[0.16, 0.15, 0.17, 0.04])
    ##############################################
    # Write labels
    xylabels(ax)
    ##############################################
    # Draw positions of galaxies
    # draw_galaxies(fig, ax)
    ##############################################
    # Save the figure
    plt.savefig("figs/sn.pdf", dpi=100)
    plt.savefig("figs/sn.eps", dpi=2500, format="eps")
    plt.savefig("figs/sn.png", dpi=300)
    return

def draw_galaxies(fig, ax):
    """ Draw galaxies in Richter 1987 catalog. """
    table = os.path.join(tables_dir, "misgeld_et_al_2008.tsv")
    ra, dec, diam = np.loadtxt(table, usecols=(0,1,15), delimiter="|").T
    ################################################
    # Center is set in NGC 3311 according to catalog
    x = canvas.arcsec2kpc(3600. * (ra - canvas.ra0))
    y = canvas.arcsec2kpc(3600. * (dec -canvas.dec0))
    #################################################
    ax.plot(x,y,"og", ms=16, markerfacecolor='none', mec="y")
    return

def find_chart():
    """ Produces a map of the signal to noise per bin according to pPXF. """
    ###############################################
    # Read values of S/N
    sn = np.loadtxt(outtable, usecols=(14,))
    xs,ys = np.loadtxt(outtable, usecols=(1,2)).T
    specs = np.loadtxt(outtable, usecols=(0,), dtype=str)
    ###############################################
    # Find good (and bad) regions according to S/N
    good = np.where(((~np.isnan(sn)) & (sn>=sn_cut)))[0]
    bad = np.where((sn<sn_cut))[0]
    ###############################################
    # Filter arrays for S/N
    sn = sn[good]
    xs = xs[good]
    ys = ys[good]
    specs = specs[good].tolist()
    specs = [x.replace(".fits", "")[1:] for x in specs]
    ###############################################
    # Set limits for the plot
    norm = Normalize(0, 1)
    ###############################################
    # Set colormap
    # cmap = brewer2mpl.get_map('YlGnBu', 'sequential', 5).mpl_colormap
    # Produces a collection of polygons with colors according to S/N values
    coll = PolyCollection(polygons_bins[good], array=np.ones_like(sn),
                          cmap="gray", edgecolors='0.5', norm=norm)
    ###############################################
    # Initiate figure and axis for matplotlib
    fig = plt.figure(figsize=(6.25,6))
    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.08, right=0.985, bottom = 0.08, top=0.985, hspace = 0.05,
               wspace=0.06)
    ax = plt.subplot(gs[0])
    ###############################################
    # Draw the polygons
    draw_map(fig, ax, coll)
    ###############################################
    # Add contours according to V-band image
    draw_contours("vband", fig, ax)
    ###############################################
    for x,y,spec in zip(xs, ys, specs):
        ax.text(x, y, spec, fontsize=10)
    # Write labels
    xylabels(ax)
    ##############################################
    # Save the figure
    plt.show()
    plt.savefig("figs/find_chart.pdf")
    return

def make_kin_summary(loess=False, contours="vband", format="png",
                     sn_lims=None, sn_loess=None, sn_sig=True):
    """ Make maps for the Lick indices in a single panel. """
    ##########################################################################
    # Set the limits for the S/N in case it is not defined by the user
    # sn_cut is defined in the setup_n3311 file
    if sn_lims == None:
        sn_lims = [sn_cut] * 4
    ##########################################################################
    # In case of LOESS smoothing, set the smoothing region (S/N < sn_loess)
    if sn_loess == None:
        sn_loess = [25,25,25,25]
    ##########################################################################
    # Read data values for Lick indices
    data = np.loadtxt(outtable, usecols=(5,7,9,11)).T
    # Read spectra name
    s = np.genfromtxt(outtable, usecols=(0,), dtype=None).tolist()
    ########################################################
    # Read coords and S/N
    xall, yall, sn = np.loadtxt(outtable, usecols=(1,2, 14)).T
    ##########################################################
    # If using S/N / sigma instead of S/N
    if sn_sig == True:
        sn /= data[1] / 100.
    ########################################################
    # Read values of other authors
    tab1a, tab1b = get_richtler()
    tab2 = get_ventimiglia()
    ###############################################
    # Details of the maps
    # Name to be displayed during processing
    titles = [r"velocity", r"sigma", r"h3", r"h4"]
    # Tex strings to be used in labels
    cb_label = [r"V$_{\rm LOS}$ [km/s]", r"$\sigma_{\rm LOS}$ [km/s]",
                r"$h_3$", r"$h_4$"]
    # Ranges of the plots
    lims = [[3750, 4000], [200, 500], [None, None], [None, None]]
    # Position of the colorbars
    xcb = [0.075, 0.555]
    xcb = xcb + xcb
    yc1 = 0.56
    yc2 = 0.085
    ycb = [yc1, yc1, yc2, yc2]
    # Colormap
    cmap = "cubelaw"
    ylabels = [1,0,1,0]
    xlabels = [0,0,1,1]
    cb_fmts = ["%d", "%d", "%.2f", "%.2f"]
    ###############################################
    # Initialize figure and subplots
    fig = plt.figure(figsize=(12.5, 12))
    gs = gridspec.GridSpec(2,2)
    gs.update(left=0.045, right=0.988, bottom=0.05, top=0.99, hspace=0.03,
              wspace=0.03)
    # Loop for figures
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(titles[i])
        good = np.where(((~np.isnan(vector)) & (sn>sn_lims[i])))[0]
        if loess:
            sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_loess[i])))[0]
            sn_low = np.delete(good, sn_high)
            vector_low = ll.loess_2d(xall[sn_low], yall[sn_low],
                                     vector[sn_low], frac=frac_loess)
            vector_high = vector[sn_high]
            good = np.hstack((sn_high, sn_low ))
            v = np.hstack((vector_high, vector_low))
        else:
            v = vector[good]
        mad =  1.4826 * np.median(np.abs(v - np.median(v)))
        ######################################################################
        # Set limits according to median deviation if not defined in lims
        vmin = np.median(v) - 1.5 * mad if lims[i][0] == None else lims[i][0]
        vmax = np.median(v) + 1.5 * mad if lims[i][0] == None else lims[i][1]
        norm = Normalize(vmin=vmin, vmax=vmax)
        ######################################################################
        ax = plt.subplot(gs[i])
        coll = PolyCollection(polygons_bins[good], array=v, cmap=cmap,
                              edgecolors='w', norm=norm)
        draw_map(fig, ax, coll)
        draw_contours(contours, fig, ax)
        plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=1000,
                            color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                  cbar_pos=[xcb[i], ycb[i], 0.09, 0.02],
                  ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i],
                  labelsize=12)
        xylabels(ax, y=ylabels[i], x=xlabels[i])
        if i not in [0,2]:
            ax.set_yticklabels([])
        if i < 2:
           ax.set_xticklabels([])
        #####################################################
        # Draw long slits of other papers
        #####################################################
        if i > 1:
            continue
        for tab in [tab1a, tab1b, tab2]:
            norm = Normalize(vmin=lims[i][0], vmax=lims[i][1])
            points = np.array([tab[:,0], tab[:,1]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]],
                                      axis=1)
            lc = LineCollection(segments, array=tab[:,i+2],
                            cmap=cmap, norm=norm, linewidth=10, zorder=800,
                            edgecolor="w")
            ax.add_collection(lc)
    nloess = "_loess" if loess else ""
    plt.savefig("figs/kinmaps{0}_{1}.{2}".format(nloess, contours, format),
                dpi=100)
    return
        
def draw_map(fig, ax, coll, bgcolor="white", lims=40):
    """ Draws a collection of rectangles in a given figure/axis. """
    ax.set_axis_bgcolor(bgcolor)
    ax.set_aspect("equal")
    ax.add_collection(coll)
    ax.minorticks_on()
    ax.set_xlim([lims, -lims])
    ax.set_ylim([-lims, lims])
    return

def draw_colorbar(fig, ax, coll, ticks=None, cblabel="", cbar_pos=None, 
                  cb_fmt="%i", labelsize=12, pm=False):
    """ Draws the colorbar in a figure. """
    if cbar_pos is None:
        cbar_pos=[0.14, 0.13, 0.17, 0.04]
    cbaxes = fig.add_axes(cbar_pos)
    cbar = plt.colorbar(coll, cax=cbaxes, orientation='horizontal',
                        format=cb_fmt)
    cbar.set_ticks(ticks)
    cbar.ax.set_xlabel(cblabel)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.set_label(cblabel,size=labelsize)
    if pm:
        newticks = []
        for i,l in enumerate(cbar.ax.get_xticklabels()):
            label = str(l)[10:-2]
            if i == 0:
                newticks.append(r"$\leq${0}".format(label))
            elif i+1 == len(cbar.ax.get_xticklabels()):
                newticks.append(r"$\geq${0}".format(label))
            else:
                newticks.append(r"{0}".format(label))
        cbar.ax.set_xticklabels(newticks)
    cl = plt.getp(cbar.ax, 'xmajorticklabels')
    plt.setp(cl, fontsize=10)     
    return

def xylabels(ax, x=True, y=True):
    if x:
        ax.set_xlabel("X [kpc]")
    if y:
        ax.set_ylabel("Y [kpc]")
    return

def draw_contours(im, fig, ax, c="k", label=True):
    """ Draw the contours of the V-band or residual image. """
    if im == "residual":
        contours = np.linspace(22, 25, 4)
        contours2 = np.linspace(22.5, 25.5, 4)
        datasmooth = canvas_res.data_smooth
        extent = canvas_res.extent
    elif im == "vband":
        contours = np.linspace(19,23,5)
        contours2 = np.linspace(19.5,23.5,5)
        datasmooth = canvas.data_smooth
        extent = canvas.extent
    elif im == "xrays":
        contours = np.linspace(90,200,8)
        contours2 = contours
        datasmooth = canvas_xrays.data_smooth
        extent = canvas_xrays.extent
    cs = ax.contour(datasmooth, contours, 
               extent=extent, colors=c)
    ax.contour(datasmooth, contours2, 
               extent=extent, colors=c, zorder=900)
    if label:
        plt.clabel(cs, inline=1, fontsize=8, fmt='%.1f')

def make_kinematics():
    """ Make maps for kinematis individually. """
    # Read data values for vel, sigma, h3, h4
    data = np.loadtxt(outtable, usecols=(5,7,9,11)).T
    xall, yall, sn = np.loadtxt(outtable, usecols=(1,2,14,)).T
    ###############################################
    # Details of the maps
    names = [r"vel", r"sigma", r"h3", r"h4"]
    cb_label = [r"V$_{\rm LOS}$ (km/s)", r"$\sigma_{\rm LOS}$ (km/s)",
                r"$h_3$", r"$h_4$"]
    lims = [[3750,4000], [150,500], [-0.08, 0.08], [-0.15, 0.15] ]
    xcb = [0.068, 0.385, 0.705]
    ###############################################
    # Set the threshold S/N for smoothing
    # Higher values than this values are not smoothed
    sn_thres = [25., 25., 1000, 1000]
    ###############################################
    # Read values of other authors
    tab1a, tab1b = get_richtler()
    tab2 = get_ventimiglia()
    ###############################################
    # Loop for figures
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(names[i])
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_thres[i])))[0]
        sn_low = np.delete(good, sn_high)
        vector_low = ll.loess_2d(xall[sn_low], yall[sn_low], vector[sn_low],
                                 frac=frac_loess)
        vector_high = vector[sn_high]
        good = np.hstack((sn_high, sn_low ))
        v_loess = np.hstack((vector_high, vector_low))
        v = vector[good]
        vmin = lims[i][0] if lims[i][0] else v_loess.min()
        vmax = lims[i][1] if lims[i][1] else v_loess.max()
        fig = plt.figure(figsize=(15, 5.1))
        gs = gridspec.GridSpec(1,3)
        gs.update(left=0.051, right=0.985, bottom=0.11, top=0.975, hspace=0.06,
                  wspace=0.06)
        vs = [v, v_loess, v_loess]
        ylabels = [1,0,0]
        contours = ["vband", "vband", "residual"]
        cb_fmts=["%i","%i", "%.2f", "%.2f"]
        ####################################################
        # Produces pannels
        ####################################################
        for j in range(3):
            ax = plt.subplot(gs[j])
            norm = Normalize(vmin=vmin, vmax=vmax)
            coll = PolyCollection(polygons_bins[good], array=vs[j],
                                  cmap="Spectral_r",
                                  edgecolors='w', norm=norm, linewidths=0.4)
            draw_map(fig, ax, coll)
            draw_contours(contours[j], fig, ax)
            plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10000,
                                color="w"))
            draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[xcb[j], 0.18, 0.08, 0.04],
                      ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i])
            xylabels(ax, y=ylabels[j])
            if j > 0:
                ax.set_yticklabels([])
            #####################################################
            # Draw long slits of other papers
            #####################################################
            if i > 1:
                continue
            for tab in [tab1a, tab1b, tab2]:
                norm = Normalize(vmin=vmin, vmax=vmax)
                points = np.array([tab[:,0], tab[:,1]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]],
                                          axis=1)
                lc = LineCollection(segments, array=tab[:,i+2],
                                cmap="cubelaw", norm=norm, linewidth=5)
                ax.add_collection(lc)
        plt.savefig("figs/{0}.pdf".format(names[i]))
        plt.savefig("figs/{0}.png".format(names[i]))
        plt.savefig("figs/{0}.eps".format(names[i]), fmt="eps")

if __name__ == "__main__":
    plt.ioff()
    ####################################################
    # Set the fraction to be used in the smoothing maps
    frac_loess = 0.3
    ####################################################
    # Set the name of the table after merging tables
    ####################################################
    outtable = "results_halo_h007.dat"
    # Set the background images for contours
    canvas = cv.CanvasImage("vband")
    canvas.data_smooth = ndimage.gaussian_filter(canvas.data, 3, order=0.)
    canvas_res = cv.CanvasImage("residual")
    canvas_res.data_smooth = ndimage.gaussian_filter(canvas_res.data, 5, 
                                                     order=0.)
    canvas_xrays = cv.CanvasImage("xrays")
    canvas_xrays.data_smooth = ndimage.gaussian_filter(canvas_xrays.data, 0.8,
                                                     order=0.)
    ####################################################
    # Switch to data folder
    workdir = os.path.join(home, "single2")
    os.chdir(workdir)
    ####################################################
    # Create folder for output files
    ####################################################
    if not os.path.exists("figs"):
        os.mkdir("figs")
    #####################################################################
    # Produces the final table
    #####################################################################
    merge_tables()
    #######################################################
    # Use canvas properties to retrieve the id of the slits
    # Set slits to be used: 0-sky 1-halo 2-point sources, 3-HCC007
    #######################################################
    slits = sorted([y for x,y in zip(canvas.slits.type, canvas.slits.ids)
             if x in [1,3]])
    # Remove slits in the chip gaps
    ignore_slits = ["cen2_s24", "inn2_s21", "cen1_s22",
                    "inn1_s22", "inn2_s27", "inn1s35"]
    slits = [x for x in slits if x not in ignore_slits]
   ####################################################
    # Positions of the slits
    xy = get_positions_by_slits(slits)
    ###############################################################
    # Create polygons
    ###############################################################
    polygons = make_voronoi(xy)
    specs = np.loadtxt(outtable, usecols=(0,), dtype=str).tolist()
    s = [x.split("n3311")[1].replace(".fits", "") for x in specs]
    idx = [slits.index(x) for x in s if x in slits]
    polygons_bins =  polygons[idx]
    #####################################################################
    # Produces the final table
    #####################################################################
    merge_tables()
    ####################################################
    #Make find chart
    ####################################################
    # find_chart()
    ####################################################
    # Produce a map with the S/N according to pPXF table
    # make_sn()
    ####################################################
    # Produce maps for all moments
    make_kinematics()
    # make_kin_summary(loess=0, contours="vband", format="png",
    #                  sn_lims=[0.2,1.,2,2], sn_loess=[20,20,1000,1000],
    #                 sn_sig=True)

