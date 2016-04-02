# -*- coding: utf-8 -*-
"""
Created on Thu May 8 15:44:23 2014

Produce 2d maps of all measured and modeled properties of the Hydra cluster 
using the binned spectra.

@author: cbarbosa
"""

import os
import sys

import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from matplotlib.collections import LineCollection
from scipy.interpolate import griddata
from matplotlib import cm
import brewer2mpl


from config import *
import newcolorbars as nc
import canvas as cv
import cap_loess_2d as ll
from voronoi_polygons import voronoi_polygons

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
        slit = spec.split(".")[0].split("_", 1)[1][5:] 
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

def merge_polys():
    """ Merge polygons to new binning. """
    specs = np.loadtxt("binning.txt", usecols=(0,), dtype=str).tolist()
    bins = np.loadtxt("binning.txt", usecols=(1,))
    specs = [s.split("_", 1)[1][5:-5] for s in specs]
    newpolys = []
    for i in range(int(bins.max()) + 1):
        indices = np.where(bins == i)[0]
        ss = [specs[j] for j in indices]
        idx = [slits.index(z) for z in ss]
        polybin = [Polygon(polygons[j]) for j in idx]
        u = cascaded_union(polybin)
        ptype = str(u.boundary)
        if ptype.startswith("LINESTRING"):
            coords = str(u.boundary).split("(")[1][:-2].split(",")
        else:
            coords = str(u.boundary.convex_hull).split("((")[1][:-2].split(",")
        coords = [[float(y) for y in x.strip().split()] 
                           for x in coords]
        coords = np.array([(x,y) for (x,y) in coords])
        newpolys.append(coords)
    return np.array(newpolys)
        
def merge_tables():
    files = ["ppxf_results.dat", "lick_corr.tsv", "populations_thomas.txt",
             "mc_lick_nsim400.txt",
             os.path.join(tables_dir, "sb_vband_single1.txt"),
             os.path.join(tables_dir, "sb_res_single1.txt")]
    s1 = np.genfromtxt(files[0], usecols=(0,), dtype=None).tolist()
    s2 = np.genfromtxt(files[1], usecols=(0,), dtype=None).tolist()
    s3 = np.genfromtxt(files[2], usecols=(0,), dtype=None).tolist()
    s4 = np.genfromtxt(files[3], usecols=(0,), dtype=None).tolist()
    s5 = np.genfromtxt(files[4], usecols=(0,), dtype=None).tolist()
    s6 = np.genfromtxt(files[5], usecols=(0,), dtype=None).tolist()
    sref = list(set(s1) & set(s2) & set(s3) & set(s4) & set(s5) & set(s6))
    ignore = ["fin1_n3311{0}.fits".format(x) for x in ignore_slits]
    sref = [x for x in sref if x not in ignore]
    if os.path.exists("template_mismatches.dat"):
        temp_mismatches = np.loadtxt("template_mismatches.dat",
                                     dtype=str).tolist()
        sref = [x for x in sref if x not in temp_mismatches]
    sref.sort()
    # if workdir in [data_dir, minus_1pc_dir, plus_1pc_dir, best_dir,
    #                rerun_dir]:
    x,y = get_positions(sref).T
    coords = get_coords(sref)
    # elif workdir == binning_dir:
    #     sref = ["s{0}.fits".format(i) for i in range(500)
    #              if "s{0}.fits".format(i) in sref]
#        sref = ["s{0}_v3800.fits".format(i) for i in range(500)
#                 if "s{0}_v3800.fits".format(i) in sref]
#         coords = np.loadtxt("spec_xy.txt", usecols=(3,4))
#         x, y = coords.T
    r = np.sqrt(x*x + y*y)
    pa = np.rad2deg(np.arctan2(x, y))      
    data1 = np.loadtxt(files[0], usecols=np.arange(1,11))
    c = 299792.458
    ##########################################################################
    # Account for difference in resolution
    # Not used anymore because the resolution is now matched in pPXF
    # fwhm_dif = (2.5 - 2.1) * c / 5500. / 2.3548
    # data1[:,2] = np.sqrt(data1[:,2]**2 - fwhm_dif**2)
    ##########################################################################
    # Loading files
    data2 = np.loadtxt(files[1], usecols=np.arange(1,26))
    data3 = np.loadtxt(files[2], usecols=(1,2,3,5,6,7,9,10,11))
    data4 = np.loadtxt(files[3], usecols=np.arange(1,26))
    data5 = np.loadtxt(files[4], usecols=(1,))
    data6 = np.loadtxt(files[5], usecols=(1,))
    ##########################################################################
    # Homogenization of the data
    data1 = match_data(s1, sref, data1)
    data2 = match_data(s2, sref, data2)
    data3 = match_data(s3, sref, data3)
    data4 = match_data(s4, sref, data4)
    data5 = match_data(s5, sref, data5)
    data6 = match_data(s6, sref, data6)
    ##########################################################################
    # Calculating composite indices: <Fe>, [MgFe]' and Mg b / <Fe>
    data24 = np.zeros_like(np.column_stack((data2, data4)))    
    for i in np.arange(25):
        data24[:, 2*i] = data2[:,i]
        data24[:, 2*i+1] = data4[:,i] 
    fe5270 = data2[:,17]
    fe5270_e = data4[:,17]
    fe5335 = data2[:,18]
    fe5335_e = data4[:,18]
    mgb = data2[:,16]
    mgb_e = data4[:,16]
    meanfe = 0.5 * (fe5270 + fe5335)
    meanfeerr = 0.5 * np.sqrt(fe5270_e**2 + fe5335_e**2)
    term = (0.72 * fe5270 + 0.28 * fe5335)
    mgfeprime = np.sqrt(mgb *  term)
    mgfeprimeerr = 0.5 * np.sqrt(term /  mgb * (mgb_e**2) +
    mgb / term * ((0.72 * fe5270_e)**2 + (0.28 * fe5335_e)**2))
    sref = np.array(sref)
    mgb_meanfe = mgb/meanfe
    mgb_meanfe[mgb_meanfe>10] = np.nan
    mgb_meanfe_err = np.sqrt((mgb * meanfeerr / meanfe**2)**2 + 
                              (mgb_e/meanfe)**2)
    ##########################################################################
    # Calculating [Fe / H]
    ##########################################################################
    feh = np.zeros((data3.shape[0], 3))
    feh[:,0] = data3[:,3] - 0.94 * data3[:,6]
    feh[:,1] = data3[:,4] -0.94 * data3[:,8]
    feh[:,2] = data3[:,5] - 0.94 * data3[:,7]
    ##########################################################################
    # Saving results
    results = np.column_stack((sref, x, y, r, pa, data1, data24, meanfe, 
              meanfeerr, mgfeprime, mgfeprimeerr, data3, coords, 
              mgb_meanfe, mgb_meanfe_err, data5, data6, feh))
    header = ['FILE', "X[kpc]", "Y[kpc]", "R[kpc]", "PA", 'V', 'dV', 'S', 'dS',
              'h3', 'dh3', 'h4', 'dh4',  'chi/DOF', 'S/N', 'Hd_A', 'dHd_A',
              'Hd_F', 'dHd_F', 'CN_1', 'dCN_1', 'CN_2', 'dCN_2', 'Ca4227',
              'dCa4227', 'G4300', 'dG4300', 'Hg_A', 'dHg_A', 'Hg_F', 'dHg_F',
              'Fe4383', 'dFe4383', 'Ca4455', 'dCa4455', 'Fe4531',  'dFe4531',
              'C4668', 'dC4668', 'H_beta', 'dH_beta', 'Fe5015', 'dFe5015',
              'Mg_1', 'dMg_1', 'Mg_2',  'dMg_2', 'Mg_b',  'dMg_b', 'Fe5270',
              'dFe5270', 'Fe5335', 'dFe5335', 'Fe5406',  'dFe5406', 'Fe5709',
              'dFe5709', 'Fe5782', 'dFe5782', 'Na_D', 'dNa_D', 'TiO_1',
              'dTiO_1', 'TiO_2',  'dTiO_2', "<Fe>", "d<Fe>", "[MgFe]'",
              "d[MgFe]'", 'Age(Gyr)', 'Age-', 'Age+', '[Z/H]', '[Z/H]-',
              '[Z/H]+', '[alpha/Fe]', '[alpha/Fe]-', '[alpha/Fe]+', "RA",
              "DEC", "Mg b / <Fe>", "d Mg b / <Fe>",
              "V-band surface brightness (mag arcsec-2)",
              "Residual V-band surface brightness (mag arcsec-2)",
              "[Fe / H]", "[Fe / H] lower limit", "[Fe / H] upper limit"]
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

def make_ages_metal_alpha():
    data = np.loadtxt("results.tab", usecols=(69,72,75)).T
    x, y, sn = np.loadtxt("results.tab", usecols=(1,2,14)).T
    names = [r"age", r"metal", "alpha"]
    cb_label = [r"Age [Gyr]", r"[Z/H]", r"[$\alpha$/Fe]"]
    cmaps = [brewer2mpl.get_map('YlOrRd', 'sequential', 9).mpl_colormap,
             brewer2mpl.get_map('YlGnBu', 'sequential', 9).mpl_colormap,
             brewer2mpl.get_map('YlGn', 'sequential', 9).mpl_colormap]
    lims = [[8,15], [-.2,0.4], [.1,.5]]
    for i, vector in enumerate(data):
        sys.stdout.write("Generating map for {0}: ".format(names[i]))
        sys.stdout.flush()
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        vector1 = vector[good]
        vector2 = ll.loess_2d(x[good], y[good], vector[good], frac=frac_loess)
        names_app = ["", "_loess"]
        for k, v in enumerate([vector1, vector2]):
            sys.stdout.write("{0} ".format(str(k+1)))
            sys.stdout.flush()
            vmin = lims[i][0] if lims[i][0] else vector2.min()
            vmax = lims[i][1] if lims[i][0] else vector2.max()
            norm = Normalize(vmin=vmin, vmax=vmax)
            coll = PolyCollection(polygons_bins[good], array=v, cmap=cmaps[i],
                                  edgecolors='none', norm=norm)
            bg = PolyCollection(polygons, array=np.zeros(len(polygons)), 
                                cmap="gray_r", edgecolors='none') 
            draw_map(coll, bg, names[i]+names_app[k], cb_label[i], bgcol="0.9") 
        sys.stdout.write("Done!\n")
        sys.stdout.flush()  
    return 
    
def make_stellar_populations_horizontal():
    """ Deprecated version of panels for stellar populations. """
    data = np.loadtxt("results.tab", usecols=(69,72,75)).T
    x, y, sn = np.loadtxt("results.tab", usecols=(1,2,14)).T
    names = [r"age", r"metal", "alpha"]
    cb_label = [r"Age [Gyr]", r"[Z/H]", r"[$\alpha$/Fe]"]
    cmaps = [cm.get_cmap("Reds"),cm.get_cmap("Blues"), cm.get_cmap("Greens")]
    cmaps = [nc.cmap_map(lambda x: x*0.5 + 0.45, x) for x in cmaps]
    lims = [[8,15], [-.2,0.4], [.1,.5]]
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1,3)
    gs.update(left=0.051, right=0.985, bottom = 0.09, top=0.98, hspace = 0.06,
               wspace=0.06)
    xcb = [0.068, 0.385, 0.7]        
    for i, ss in enumerate(gs):
        vector = data[i]
        ax = plt.subplot(ss)
        ax.set_xlim([40, -40])
        ax.set_ylim([-40, 40])
#        ax.set_axis_bgcolor("0.9")
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        vector1 = vector[good]
        Z = ll.loess_2d(x[good], y[good], vector[good], frac=frac_loess)
        vmin = lims[i][0] if lims[i][0] else Z.min()
        vmax = lims[i][1] if lims[i][0] else Z.max()
        norm = Normalize(vmin=vmin, vmax=vmax)
        coll = PolyCollection(polygons_bins[good], array=Z, cmap=cmaps[i],
                                  edgecolors='none', norm=norm)
        im = ax.add_collection(coll)
        ax.minorticks_on()
        if i == 0:         
            ax.set_ylabel("Y [kpc]")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("X [kpc]")
        datasmooth = ndimage.gaussian_filter(canvas.data, nsig, order=0.)       
        plt.minorticks_on()
        cbaxes = fig.add_axes([xcb[i], 0.14,.1, 0.035])
        cbar = plt.colorbar(coll, cax=cbaxes, 
                           orientation='horizontal', format='%.1f')
        cbar.set_ticks(np.linspace(vmin, vmax, 5))
        cbar.ax.set_xlabel(cb_label[i]) 
        cbar.ax.xaxis.set_label_position('top')
        cbar.ax.xaxis.set_ticks_position('bottom')
        cl = plt.getp(cbar.ax, 'xmajorticklabels')
        plt.setp(cl, fontsize=9)
        cs = ax.contour(datasmooth, contours, 
                   extent=canvas.extent, colors="k",
                    linewidths=0.8)
        ax.contour(datasmooth, contours2, 
                   extent=canvas.extent, colors="k",
                   linewidths=0.8)
        plt.clabel(cs, inline=1, fontsize=8, fmt='%.1f')
    filename = "figs/panel_stpop.pdf"
    plt.savefig(filename)
    plt.savefig(filename.replace(".pdf", ".png")) 
    return
    
def make_sn():
    """ Produces a map of the signal to noise per bin according to pPXF. """
    ###############################################
    # Read values of S/N
    sn = np.loadtxt(outtable, usecols=(14,))
    ###############################################
    # Find good (and bad) regions according to S/N
    good = np.where(((~np.isnan(sn)) & (sn>=sn_cut)))[0]
    bad = np.where((sn<sn_cut))[0]
    ###############################################
    # Filter S/N 
    sn = sn[good]
    ###############################################
    # Colorbar limits
    vmin, vmax = 10, 50
    # Set limits for the plot
    norm = Normalize(vmin, vmax)
    ###############################################
    # Set colormap
    cmap = brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colormap
    # cmap = nc.cmap_map(lambda x: x*0.7 + 0.23, cm.get_cmap("cubehelix"))
    # cmap = nc.cmap_discretize(cmap, 8)
    cmap = cm.get_cmap("cubelaw_r")
    cmap = nc.cmap_discretize(cmap, 8)
    # cmap = "Spectral"
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

def draw_galaxies(fig, ax, write=False):
    """ Draw galaxies in Richter 1987 catalog. """
    table = os.path.join(tables_dir, "misgeld_et_al_2008.tsv")
    ra, dec, diam = np.loadtxt(table, usecols=(0,1,15), delimiter="|").T
    ids = np.loadtxt(table, usecols=(2,), delimiter="|", dtype=str).tolist()
    ids = [r"$H{0}$".format(x) for x in ids]
    ################################################
    # Center is set in NGC 3311 according to catalog
    x = canvas.arcsec2kpc(3600. * (ra - canvas.ra0))
    y = canvas.arcsec2kpc(3600. * (dec -canvas.dec0))
    #################################################
    ax.plot(x,y,"og", ms=16, markerfacecolor='none', mec="r", mew=2)
    if not write:
        return
    for px, py, gal in zip(x,y,ids):
        if np.abs(px) > 38 or np.abs(py) > 38:
            continue
        ax.scatter(px+1,py+4,marker=gal, c="r", s=500, edgecolors='r',
                   zorder=100000)
    return

def make_sb(im="vband"):
    """ Produces a map of the surface brightness. """
    ###############################################
    # Read values of S/N
    if im == "vband":
        datasmooth = ndimage.gaussian_filter(canvas.data, 5, order=0.)
        # Colorbar limits
        vmin, vmax = 20,25
    else:
        datasmooth = ndimage.gaussian_filter(canvas_res.data, 5, order=0.)
        # Colorbar limits
        vmin, vmax = 22,26
    x,y = np.loadtxt("results.tab", usecols=(1,2)).T
    specs = np.loadtxt("results.tab", usecols=(0,), dtype=str)
    x0, x1, y0, y1 = canvas.extent
    ysize, xsize = canvas.data.shape
    ypix =  (y-y0)/(y1-y0) * ysize
    xpix = (x-x0)/(x1-x0) * xsize
    ##########################################################################
    # Calculate SB for Lodo spectra
    rl, pal = np.loadtxt(os.path.join(tables_dir, "coccato2011_indices.tsv"),
                         usecols=(1,2)).T
    xl, yl = polar2cart(rl, pal)
    xl = canvas.arcsec2kpc(xl)
    yl = canvas.arcsec2kpc(yl)
    ypixl =  (yl-y0)/(y1-y0) * ysize
    xpixl = (xl-x0)/(x1-x0) * xsize
    for xx,yy in zip(xpixl, ypixl):
        print xx, yy,
        try:
            print datasmooth[yy,xx]
        except:
            print
    ##########################################################################
    sb = []
    for xx,yy in zip(xpix, ypix):
        sb.append(datasmooth[yy,xx])
    sb = np.array(sb)
    # Set limits for the plot
    norm = Normalize(vmin, vmax)
    ###############################################
    # Set colormap
    cmap = cm.get_cmap("hot_r")
    cmap = nc.cmap_map(lambda x: x*0.5 + 0.3, cm.get_cmap("hot_r"))
    # Produces a collection of polygons with colors according to S/N values
    coll = PolyCollection(polygons_bins, array=sb, cmap=cmap,
                          edgecolors='none', norm=norm)
    ###############################################
    # Initiate figure and axis for matplotlib
    fig = plt.figure(figsize=(6.4,6))
    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.08, right=0.985, bottom = 0.08, top=0.985, hspace = 0.05,
               wspace=0.06)
    ax = plt.subplot(gs[0])
    ###############################################
    # Draw the polygons
    draw_map(fig, ax, coll)
    ###############################################
    # Add contours according to V-band image
    draw_contours("vband", fig, ax, fc="k")
    # Draw actual slit positions
    # canvas.draw_slits(ax, slit_type=1, c="r" )
    # canvas.draw_slits(ax, slit_type=3, c="r" )
    ###############################################
    # Draw white rectangle in the position of the colorbar so background
    # stars do not overplot the labels and ticks
    plt.gca().add_patch(Rectangle((20,-38),18,10, alpha=1, zorder=10,
                        color="w"))
    ###############################################
    # Draw the colorbar
    draw_colorbar(fig, ax, coll, ticks=np.linspace(vmin,vmax,5),
                  cblabel=r"S/N per pixel")
    ##############################################
    # Write labels
    xylabels(ax)
    ##############################################
    sb = np.column_stack((specs, sb))
    # Save the figure
    if im == "vband":
        plt.savefig("figs/vband.png", dpi=100)
        with open(os.path.join(tables_dir, "sb_vband_binning_20.txt"), "w") as f:
            np.savetxt(f, sb, fmt="%s")
    else:
        plt.savefig("figs/res.png", dpi=100)
        with open(os.path.join(tables_dir, "sb_res_binning_20.txt"), "w") as f:
            np.savetxt(f, sb, fmt="%s")
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
    # draw_contours("vband", fig, ax)
    ###############################################
    for x,y,spec in zip(xs, ys, specs):
        ax.text(x, y, spec, fontsize=10)
    # Write labels
    xylabels(ax)
    ##############################################
    # Save the figure
    plt.savefig("figs/find_chart.pdf")
    return

def make_kinematics():
    """ Make maps for kinematis individually. """
    # Read data values for vel, sigma, h3, h4
    data = np.loadtxt(outtable, usecols=(5,7,9,11)).T 
    xall, yall, sn = np.loadtxt("results.tab", usecols=(1,2,14,)).T
    ###############################################
    # Details of the maps
    names = [r"vel", r"sigma", r"h3", r"h4"]
    cb_label = [r"V$_{\rm LOS}$ [km/s]", r"$\sigma_{\rm LOS}$ [km/s]", 
                r"$h_3$", r"$h_4$"]
    lims = [[3750,4000], [150,500], [None, None], [None, 0.15] ]
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
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1,3)
        gs.update(left=0.051, right=0.985, bottom=0.10, top=0.98, hspace=0.06,
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
                                  cmap="cubelaw", 
                                  edgecolors='none', norm=norm)
            draw_map(fig, ax, coll)
            draw_contours(contours[j], fig, ax)
            plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10, 
                                color="w"))
            draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[xcb[j], 0.16, 0.09, 0.04],
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

def make_kin_summary(loess=True):
    """ Make single panel for 4 LOSVD moments maps. """
    ########################################################
    # Read data values for Lick indices
    data = np.loadtxt(outtable, usecols=(5,7,9,11)).T
    # Read spectra name
    s = np.genfromtxt(outtable, usecols=(0,), dtype=None).tolist()
    ########################################################
    # Read coords and S/N
    xall, yall, sn = np.loadtxt(outtable, usecols=(1,2, 14)).T
    ########################################################
    # Read values of other authors
    tab1a, tab1b = get_richtler()
    tab2 = get_ventimiglia()
    ###############################################
    # Details of the maps
    titles = [r"velocity", r"sigma",
                r"h3", r"h4"]
    cb_label = [r"V$_{\rm LOS}$ [km/s]", r"$\sigma_{\rm LOS}$ [km/s]",
                r"$h_3$", r"$h_4$"]
    lims = [[3800, 4000], [150, 400], [-.08, 0.08], [-0.05, .15],
             [None, None], [None, None], [None, None], [None, None],
             [None, None]]
    xcb = [0.09, 0.56]
    xcb = xcb + xcb
    yc1 = 0.56
    yc2 = 0.085
    ycb = [yc1, yc1, yc2, yc2]
    cmap = brewer2mpl.get_map('Reds', 'sequential', 9).mpl_colormap
    cmap = "cubelaw"
    # cmap = nc.cmap_map(lambda x: x*0.5 + 0.45, cmap)
    fig = plt.figure(figsize=(11.5, 11))
    gs = gridspec.GridSpec(2,2)
    gs.update(left=0.06, right=0.988, bottom=0.05, top=0.99, hspace=0.03,
              wspace=0.03)
    ylabels = [1,0,1,0]
    xlabels = [0,0,1,1]
    sn_thres = 25
    cb_fmts = ["%d", "%d", "%.2f", "%.2f"]
    ###############################################
    # Loop for figures
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(titles[i])
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        v = vector[good]
        robust_sigma =  1.4826 * np.median(np.abs(v - np.median(v)))
        vmin = np.median(v) - 1.5 * robust_sigma
        vmax = np.median(v) + 1.5 * robust_sigma
        sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_thres)))[0]
        sn_low = np.delete(good, sn_high)
        vector_low = ll.loess_2d(xall[sn_low], yall[sn_low], vector[sn_low],
                                 frac=frac_loess)
        vector_high = vector[sn_high]
        good = np.hstack((sn_high, sn_low ))
        if loess:
            v = np.hstack((vector_high, vector_low))
        else:
            v = vector[good]
        ax = plt.subplot(gs[i])
        norm = Normalize(vmin=lims[i][0], vmax=lims[i][1])
        coll = PolyCollection(polygons_bins[good], array=v, cmap=cmap,
                              edgecolors='w', norm=norm)
        draw_map(fig, ax, coll)
        draw_contours("vband", fig, ax)
        plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10,
                            color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                  cbar_pos=[xcb[i], ycb[i], 0.13, 0.02],
                  ticks=np.linspace(lims[i][0], lims[i][1], 4), cb_fmt=cb_fmts[i],
                  labelsize=24)
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
                            cmap="cubelaw", norm=norm, linewidth=5)
            ax.add_collection(lc)
    nloess = "_loess" if loess else ""
    plt.savefig("figs/kinmaps{0}.png".format(nloess), dpi=250)
    return

def make_lick():
    """ Make maps for the Lick indices individually. """
    ########################################################
    # Read data values for Lick indices
    data = np.loadtxt(outtable, 
                      usecols=(39,41,43,45,47,49,51,53,55,65,67,80)).T 
    # Read spectra name
    s = np.genfromtxt(outtable, usecols=(0,), dtype=None).tolist() 
    ########################################################
    # Read coords and S/N
    xall, yall, sn = np.loadtxt(outtable, usecols=(1,2, 14)).T
    ########################################################
    ###############################################
    # Details of the maps
    titles = ["H_beta", "Fe5015", "Mg_1", "Mg_2", "Mg_b", "Fe5270", "Fe5335", 
              "Fe5406", "Fe5709", "meanFe", "mgfeprime", "mgb_meanfe"]
    cb_label = [r"H$\beta$ [$\AA$]", r"Fe5015 [$\AA$]", r"Mg$_1$ [mag]",
                r"Mg$_2$ [mag]", r"Mg $b$ [$\AA$]", r"Fe5270 [$\AA$]",
                r"Fe5335 [$\AA$]", r"Fe5406 [$\AA$]", r"Fe5709 [$\AA$]",
                r"$<$Fe$>$ [$\AA$]", r"[MgFe]' [$\AA$]", r"Mg $b$ / $<$Fe$>$"]
    names = [r"H_beta", r"Fe5015", r"Mg_1", r"Mg_2", r"Mg_b", r"Fe5270", 
                r"Fe5335", r"Fe5406", r"Fe5709", "meanFe", "mgfeprime", 
               "mgb_meanfe" ]
    lims = [[1.,2.2], [None, None], [None, None], [None, None], 
             [1.5,6.], [None, None], [None, None], [None, None], 
             [None, None], [1.,4.], [2.5,5.], [None, None]]
    lims = [[None, None], [None, None], [None, None], [None, None],
             [None, None], [None, None], [None, None], [None, None],
             [None, None], [None, None], [None, None], [None, None]]
    xcb = [0.068, 0.385, 0.705] 
    cb_fmts=["%.2f","%.2f", "%.3f", "%.3f", "%.2f","%.2f", "%.2f","%.2f", 
             "%.2f","%.2f", "%.2f","%.2f"]
    cmap = nc.cmap_map(lambda x: x*0.5 + 0.45, cm.get_cmap("hot"))
    ###############################################
    # Set the threshold S/N for smoothing
    # Higher values than this values are not smoothed
    sn_thres = [25., 25., 25., 25., 25., 25., 25., 25., 25., 25., 25., 25.]
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
        fig = plt.figure(figsize=(15, 5))
        gs = gridspec.GridSpec(1,3)
        gs.update(left=0.051, right=0.985, bottom=0.10, top=0.98, hspace=0.06,
                  wspace=0.06) 
        vs = [v, v_loess, v_loess]
        ylabels = [1,0,0]
        contours = ["vband", "vband", "residual"]
        ####################################################
        # Produces pannels
        ####################################################
        for j in range(3):
            ax = plt.subplot(gs[j])
            norm = Normalize(vmin=vmin, vmax=vmax) 
            coll = PolyCollection(polygons_bins[good], array=vs[j], 
                                  cmap=cmap,
                                  edgecolors='none', norm=norm)
            draw_map(fig, ax, coll)
            draw_contours(contours[j], fig, ax)
            plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10, 
                                color="w"))
            draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[xcb[j], 0.16, 0.09, 0.04],
                      ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i])
            xylabels(ax, y=ylabels[j])
            if j > 0:
                ax.set_yticklabels([])
        plt.savefig("figs/{0}.pdf".format(names[i]))
        # plt.savefig("figs/{0}.eps".format(names[i]))

def make_lick2(loess=False, rlims=40):
    """ Make maps for the Lick indices in a single panel. """
    ########################################################
    # Read data values for Lick indices
    data = np.loadtxt(outtable,
                      usecols=(39,41,47,49,51,53)).T
    # Read spectra name
    s = np.genfromtxt(outtable, usecols=(0,), dtype=None).tolist()
    ########################################################
    # Read coords and S/N
    xall, yall, sn = np.loadtxt(outtable, usecols=(1,2, 14)).T
    ########################################################
    # Details of the maps
    titles = ["H_beta", "Fe5015", "Mg_b", "Fe5270", "Fe5335",
              "Fe5406", "Fe5709"]
    cb_label = [r"H$\beta$ [\AA]", r"Fe5015 [\AA]", r"Mg $b$ [\AA]",
                r"Fe5270 [\AA]", r"Fe5335 [\AA]",
                r"Fe5406 [\AA]", r"Fe5709 [\AA]"]
    lims = [[None, None], [None, None], [None, None], [None, None],
             [None, None], [None, None], [None, None], [None, None],
             [None, None]]
    xcb = [0.075, 0.39, .705]
    xcb = xcb + xcb
    yc1 = 0.575
    yc2 = 0.104
    ycb = [yc1, yc1, yc1, yc2, yc2, yc2]
    cmap = brewer2mpl.get_map('YlOrRd', 'sequential', 3).mpl_colormap
    cmap = nc.cmap_discretize(cmap, 6)
    fig = plt.figure(figsize=(13.6, 9))
    gs = gridspec.GridSpec(2,3)
    gs.update(left=0.06, right=0.985, bottom=0.055, top=0.99, hspace=0.04,
              wspace=0.075)
    ylabels = [1,0,0,1,0,0,0]
    xlabels = [0,0,0,1,1,1,1]
    sn_thres = 25
    ###############################################
    # Loop for figures
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(titles[i])
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        v = vector[good]
        robust_sigma =  1.4826 * np.median(np.abs(v - np.median(v)))
        vmin = np.median(v) - 1.0* robust_sigma
        vmax = np.median(v) + 1.0 * robust_sigma
        sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_thres)))[0]
        sn_low = np.delete(good, sn_high)
        # vector_low = ll.loess_2d(xall[sn_low], yall[sn_low], vector[sn_low],
        #                          frac=frac_loess)
        # vector_high = vector[sn_high]
        # good = np.hstack((sn_high, sn_low ))
        if loess:
            v = np.hstack((vector_high, vector_low))
        else:
            v = vector[good]
        ax = plt.subplot(gs[i])
        norm = Normalize(vmin=vmin, vmax=vmax)
        coll = PolyCollection(polygons_bins[good], array=v, cmap=cmap,
                              edgecolors='w', norm=norm)
        draw_map(fig, ax, coll, lims=rlims)
        draw_contours("vband", fig, ax)
        if rlims==40:
            plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1,
                                          zorder=10000, color="w"))
        if rlims == 20:
            plt.gca().add_patch(Rectangle((19,-19),-15,6, alpha=1,
                                          zorder=10000, color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                  cbar_pos=[xcb[i], ycb[i], 0.085, 0.02],
                  ticks=np.linspace(vmin, vmax, 4), cb_fmt="%.1f",
                  labelsize=24)
        xylabels(ax, y=ylabels[i], x=xlabels[i])
        if i not in [0,3]:
            ax.set_yticklabels([])
        if i < 3:
           ax.set_xticklabels([])
    nloess = "_loess" if loess else ""
    plt.savefig("figs/lickmaps{0}_r{1}.eps".format(nloess, rlims), dpi=1200,
                format="eps")
    return

def make_stellar_populations(loess=False, contours="vband", dwarfs=True,
                             letters=False, lims=40.):
    data = np.loadtxt(outtable, usecols=(69,72,75)).T 
    xall, yall, sn = np.loadtxt("results.tab", usecols=(1,2,14,)).T
    ###############################################
    # Details of the maps
    names = [r"age", r"metal", r"alpha"]
    cb_label = [r"Age [Gyr]", r"[Z/H]", r"[$\alpha$/Fe]"]
    cmaps = [brewer2mpl.get_map('Reds', 'sequential', 9).mpl_colormap,
             brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colormap,
             brewer2mpl.get_map('Greens', 'sequential', 9).mpl_colormap]
    cmaps = [nc.cmap_map(lambda x: x*0.54 + 0.43, x) for x in cmaps]
    cmaps = [nc.cmap_discretize(x, 6) for x in cmaps]
    xcb = [0.068, 0.385, 0.705] 
    ###############################################
    # Set the threshold S/N for smoothing
    # Higher values than this values are not smoothed
    sn_thres = [25, 25, 25]
    ###############################################
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1,3)
    gs.update(left=0.051, right=0.985, bottom=0.115, top=0.975, hspace=0.06,
              wspace=0.06)
    cb_fmts=["%.1f","%.2f", "%.2f"]
    labels = ["(D)", "(E)", "(F)"]
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
        if loess:
            v = np.hstack((vector_high, vector_low))
        else:
            v = vector[good]
        robust_sigma =  1.4826 * np.median(np.abs(v - np.median(v)))
        vmin = np.median(v) - 1.2 * robust_sigma
        vmax = np.median(v) + 1.2 * robust_sigma
        if i == 0:
            vmax = np.minimum(15., vmax)
        ylabels = [1,0,0]
        ax = plt.subplot(gs[i])
        norm = Normalize(vmin=vmin, vmax=vmax)
        coll = PolyCollection(polygons_bins[good], array=v, cmap=cmaps[i],
                                  edgecolors='none', norm=norm,
                                  linewidths=0.001)
        draw_map(fig, ax, coll, lims=lims)
        draw_contours(contours, fig, ax)
        plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10000,
                            color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[xcb[i], 0.16, 0.09, 0.04],
                      ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i])
        xylabels(ax, y=ylabels[i])
        if dwarfs:
            draw_galaxies(fig, ax)
        if i > 0:
            ax.set_yticklabels([])
        if letters:
            ax.text(0.05, 0.95, labels[i], transform=ax.transAxes,
                    fontsize=26, fontweight='bold', va='top')
    if loess:
        app = "loess"
    else:
        app = "noloess"
    plt.savefig("figs/populations_{0}_{1}_r{0}.png".format(contours, app,
                lims), dpi=300)
    return

def make_sp_panel(loess=False):
    data = np.loadtxt(outtable, usecols=(69,72,75)).T
    xall, yall, sn = np.loadtxt("results.tab", usecols=(1,2,14,)).T
    ###############################################
    # Details of the maps
    names = [r"age", r"metal", r"alpha"]
    cb_label = [r"Age [Gyr]", r"[Z/H]", r"[$\alpha$/Fe]"]
    cmaps = [cm.get_cmap("Reds"),cm.get_cmap("Blues"), cm.get_cmap("Greens")]
    cmaps = [nc.cmap_map(lambda x: x*0.5 + 0.45, x) for x in cmaps]
    xcb = [0.065, 0.38, .7]
    xcb = xcb + xcb
    yc1 = 0.56
    yc2 = 0.085
    ycb = [yc1, yc1, yc1, yc2, yc2, yc2]
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2,3)
    gs.update(left=0.045, right=0.988, bottom=0.05, top=0.99, hspace=0.04,
              wspace=0.06)
    ylabels = [1,0,0,1,0,0,0]
    xlabels = [0,0,0,1,1,1,1]
    cb_fmts=["%.1f","%.2f", "%.2f"]
    contours= ["vband", "vband", "vband", "vband", "vband",
                "vband"]
    sn_thres = 25.
    ##########################################################################
    # Produce panels
    ##########################################################################
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(names[i])
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        v = vector[good]
        sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_thres)))[0]
        sn_low = np.delete(good, sn_high)
        vector_low = ll.loess_2d(xall[sn_low], yall[sn_low], vector[sn_low],
                                 frac=frac_loess)
        vector_high = vector[sn_high]
        good = np.hstack((sn_high, sn_low ))
        v_loess = np.hstack((vector_high, vector_low))
        v = vector[good]
        robust_sigma =  1.4826*np.median(np.abs(v - np.median(v)))
        vmin = np.median(v) - 1.5 * robust_sigma
        vmax = np.median(v) + 1.5 * robust_sigma
        if i == 0:
            vmax = np.minimum(15., vmax)
        vs = [v, v_loess]
        for k,j in enumerate([i,i+3]):
            ax = plt.subplot(gs[j])
            norm = Normalize(vmin=vmin, vmax=vmax)
            coll = PolyCollection(polygons_bins[good], array=vs[k],
                                  cmap=cmaps[i], edgecolors='none', norm=norm)
            draw_map(fig, ax, coll)
            draw_contours(contours[j], fig, ax)
            plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10,
                                color="w"))
            draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[xcb[j], ycb[j], 0.085, 0.02],
                      ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i],
                      labelsize=24, pm=False)
            xylabels(ax, y=ylabels[j], x=xlabels[j])
            # draw_galaxies(fig,ax)
            if j not in [0,3]:
                ax.set_yticklabels([])
            if j < 3:
               ax.set_xticklabels([])
    loess_append = "_loess" if loess else ""
    plt.savefig("figs/populations{0}.png".format(loess_append), dpi=80)
    return

def make_sp_panel3(loess=False):
    """Draw 3 x 3 panel with Age, Z/H and E/Fe with contours of V-band,
      residuals and X-rays."""
    data = np.loadtxt(outtable, usecols=(69,72,75)).T
    xall, yall, sn = np.loadtxt("results.tab", usecols=(1,2,14,)).T
    ###############################################
    # Details of the maps
    names = [r"age", r"metal", r"alpha"]
    cb_label = [r"Age [Gyr]", r"[Z/H]", r"[$\alpha$/Fe]"]
    cmaps = [cm.get_cmap("Reds"),cm.get_cmap("Blues"), cm.get_cmap("Greens")]
    cmaps = [nc.cmap_map(lambda x: x*0.5 + 0.45, x) for x in cmaps]
    xcb = [0.065, 0.38, .7]
    xcb = xcb + xcb + xcb
    yc1 = 0.705
    yc2 = 0.39
    yc3 = 0.072
    ycb = [yc1, yc1, yc1, yc2, yc2, yc2, yc3, yc3, yc3]
    fig = plt.figure(figsize=(15, 15))
    gs = gridspec.GridSpec(3,3)
    gs.update(left=0.045, right=0.988, bottom=0.05, top=0.99, hspace=0.04,
              wspace=0.06)
    ylabels = [1,0,0,1,0,0,1,0,0]
    xlabels = [0,0,0,0,0,0,1,1,1]
    cb_fmts=["%.1f","%.2f", "%.2f"]
    contours= ["vband", "vband", "vband", "residual", "residual",
                "residual", "xrays", "xrays", "xrays"]
    sn_thres = 25.
    ##########################################################################
    # Produce panels
    ##########################################################################
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(names[i])
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        v = vector[good]
        sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_thres)))[0]
        sn_low = np.delete(good, sn_high)
        vector_low = ll.loess_2d(xall[sn_low], yall[sn_low], vector[sn_low],
                                 frac=frac_loess)
        vector_high = vector[sn_high]
        good = np.hstack((sn_high, sn_low ))
        if loess:
            v = np.hstack((vector_high, vector_low))
        else:
            v = vector[good]
        robust_sigma =  1.4826 * np.median(np.abs(v - np.median(v)))
        vmin = np.median(v) - 1. * robust_sigma
        vmax = np.median(v) + 1. * robust_sigma
        if i == 0:
            vmax = np.minimum(15., vmax)
        for j in [i,i+3,i+6]:
            ax = plt.subplot(gs[j])
            ax.set_axisbelow(True)
            norm = Normalize(vmin=vmin, vmax=vmax)
            coll = PolyCollection(polygons_bins[good], array=v, cmap=cmaps[i],
                                  edgecolors='none', norm=norm)
            draw_map(fig, ax, coll)
            draw_contours(contours[j], fig, ax)
            plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10,
                                color="w"))
            draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                  cbar_pos=[xcb[j], ycb[j], 0.085, 0.02],
                  ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i],
                  labelsize=24, pm=False)
            xylabels(ax, y=ylabels[j], x=xlabels[j])
            draw_galaxies(fig,ax)
            if j not in [0,3,6]:
                ax.set_yticklabels([])
            if j < 6:
               ax.set_xticklabels([])
    loess_append = "_loess" if loess else ""
    plt.savefig("figs/populations3{0}.png".format(loess_append), dpi=80)
    return

def make_other():
    ###############################################
    # Details of the maps
    names = [r"vband", r"residual", r"xrays"]
    cb_label = [r"$\mu_V$ (mag arcsec$^{-2}$)", r"$\mu_V$ (mag arcsec$^{-2}$)",
                r"X-rays (counts)"]
    xcb = [0.08, 0.395, 0.71]
    ###############################################
    # Set the threshold S/N for smoothing
    # Higher values than this values are not smoothed
    sn_thres = [25, 25, 25]
    ###############################################
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1,3)
    gs.update(left=0.051, right=0.985, bottom=0.11, top=0.978, hspace=0.06,
              wspace=0.06)
    cb_fmts=["%.1f","%.1f", "%d"]
    ylabels = [1,0,0]
    vmins = [20, 23., 65]
    vmaxs = [24.5, 26, 125]
    green = nc.cmap_map(lambda x: x*0.8 + 0.2, cm.get_cmap("YlGn_r"))
    cmaps = ["gray", "cubehelix", green]
    labels = ["(A)", "(B)", "(C)"]
    # canvas_xrays.data = canvas_xrays.data_smooth
    writegal = [True, False, False]
    for i, c in enumerate([canvas, canvas_res, canvas_xrays]):
        ax = plt.subplot(gs[i])
        ax.minorticks_on()
        coll = c.imshow(cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i],
                        aspect="equal")
        ax.set_xlim(40,-40)
        ax.set_ylim(-40,40)
        xylabels(ax, y=ylabels[i])
        draw_galaxies(fig, ax, write=writegal[i])
        draw_contours("vband", fig, ax, c="k")
        plt.gca().add_patch(Rectangle((6,-38),32,12, alpha=1, zorder=1000,
                            color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[xcb[i], 0.168, 0.09, 0.04], pm=0,
                      ticks=np.linspace(vmins[i], vmaxs[i], 4),
                      cb_fmt=cb_fmts[i])
        if i > 0:
            ax.set_yticklabels([])
        ax.text(0.05, 0.95, labels[i], transform=ax.transAxes,
                fontsize=26, fontweight='bold', va='top',
                bbox=dict(boxstyle="square", fc="w", ec="none"))
    plt.savefig("figs/other.png")

    return

def make_imgs():
    ###############################################
    # Details of the maps
    names = [r"vband", r"residual", r"xrays"]
    cb_label = [r"$\mu_V$ (mag arcsec$^{-2}$)", r"$\mu_V$ (mag arcsec$^{-2}$)",
                r"X-rays (counts)"]
    ###############################################
    # Set the threshold S/N for smoothing
    # Higher values than this values are not smoothed
    sn_thres = [25, 25, 25]
    ###############################################
    fig = plt.figure(1,figsize=(5.5,5))
    gs = gridspec.GridSpec(1,1)
    gs.update(left=0.08, right=0.99, bottom=0.11, top=0.978, hspace=0.06,
              wspace=0.06)
    cb_fmts=["%.1f","%.1f", "%d"]
    vmins = [20, 23., 65]
    vmaxs = [24.5, 26, 125]
    green = nc.cmap_map(lambda x: x*0.8 + 0.2, cm.get_cmap("YlGn_r"))
    cmaps = ["gray", "cubehelix", green]
    labels = ["(A)", "(B)", "(C)"]
    # canvas_xrays.data = canvas_xrays.data_smooth
    writegal = [True, False, False]
    for i, c in enumerate([canvas, canvas_res, canvas_xrays]):
        ax = plt.subplot(gs[0])
        ax.minorticks_on()
        coll = c.imshow(cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i],
                        aspect="equal")
        ax.set_xlim(40,-40)
        ax.set_ylim(-40,40)
        xylabels(ax)
        # draw_galaxies(fig, ax, write=writegal[i])
        draw_contours("vband", fig, ax, c="k")
        plt.gca().add_patch(Rectangle((5,-38),33,14, alpha=1, zorder=1000,
                            color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[0.2, 0.17, 0.25, 0.04], pm=0,
                      ticks=np.linspace(vmins[i], vmaxs[i], 4),
                      cb_fmt=cb_fmts[i])
        if i == 1:
            circle = cv.circle_xy(8.4)
            ax.plot(circle[:,0], circle[:,1], "-r", lw=3)
            ax.plot([0,0],[8.4,40], "-r", lw=3)
            ax.plot([40,8.4],[0,0], "-r", lw=3)
            t1 = ax.text(0.05, 0.85, "Off-centred\nenvelope", transform=ax.transAxes,
                    fontsize=22, fontweight='bold', va='top', color="r")
            t1.set_bbox(dict(color='w', alpha=0.8, edgecolor='none'))
            t2 = ax.text(0.25, 0.35, "Symmetric halo", transform=ax.transAxes,
                    fontsize=22, fontweight='bold', va='top', color="r")
            t2.set_bbox(dict(color='w', alpha=0.8, edgecolor='none'))
            t3 = ax.text(0.55, 0.85, "Inner Galaxy", transform=ax.transAxes,
                    fontsize=22, fontweight='bold', va='top', color="r")
            t3.set_bbox(dict(color='w', alpha=0.8, edgecolor='none'))
            ax.arrow(-8.4 * np.sqrt(2)/2, 8.4 * np.sqrt(2)/2,
                     -20 * np.sqrt(2)/2, 20 * np.sqrt(2)/2,
                    head_width=2, head_length=4, fc='r', ec='r',
                     width=0.7, zorder=1000)
        # ax.text(0.05, 0.95, labels[i], transform=ax.transAxes,
        #         fontsize=26, fontweight='bold', va='top')
        plt.savefig("figs/{0}.png".format(names[i]), dpi=120)
        plt.clf()
    return
        
def draw_map(fig, ax, coll, bgcolor="white", lims=40):
    """ Draws a collection of rectangles in a given figure/axis. """
    ax.set_aspect("equal")
    ax.add_collection(coll)
    ax.minorticks_on()
    ax.set_xlim([lims, -lims])
    ax.set_ylim([-lims, lims])
    return

def draw_colorbar(fig, ax, coll, ticks=None, cblabel="", cbar_pos=None, 
                  cb_fmt="%i", labelsize=16, pm=False):
    """ Draws the colorbar in a figure. """
    if cbar_pos is None:
        cbar_pos=[0.14, 0.13, 0.17, 0.04]
    cbaxes = fig.add_axes(cbar_pos)
    cbar = plt.colorbar(coll, cax=cbaxes, orientation='horizontal',
                        format=cb_fmt)
    cbar.set_ticks(ticks)
    cbar.ax.set_xlabel(cblabel) 
    cbar.ax.tick_params(labelsize=labelsize)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.xaxis.set_ticks_position('bottom')
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

def xylabels(ax, x=True, y=True, size=None):
    if x:
        ax.set_xlabel("X [kpc]", size=size)
    if y:
        ax.set_ylabel("Y [kpc]", size=size)
    return

def draw_contours(im, fig, ax, c="k", label=True):
    """ Draw the contours of the V-band or residual image. """
    linewidths = 1.5
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
               extent=extent, colors=c, zorder=1, linewidths=linewidths)
    cs2 = ax.contour(datasmooth, contours2,
               extent=extent, colors=c, zorder=1, linewidths=linewidths)
    if label:
        cls = ax.clabel(cs, inline=1, fontsize=8, fmt='%.1f')
        # now CLS is a list of the labels, we have to find offending ones

        # get limits if they're automatic
        xmax,xmin,ymin,ymax = plt.axis()
        ymin = -20
        Dx = xmax-xmin
        Dy = ymax-ymin

        # check which labels are near a border
        keep_labels = []
        for label in cls:
            lx,ly = label.get_position()
            if xmin <lx<xmax and ymin<ly<ymax:
                # inlier, redraw it later
                keep_labels.append((lx,ly))

        # delete the original lines, redraw manually the labels we want to keep
        # this will leave unlabelled full contour lines instead of overlapping labels

        for cline in cs.collections:
            cline.remove()
        for label in cls:
            label.remove()

        cs = ax.contour(datasmooth, contours,
               extent=extent, colors=c, zorder=1, linewidths=linewidths)
        cls = ax.clabel(cs, inline=1, fontsize=8, fmt='%.1f',
                        manual=keep_labels)
    return


def make_maps_interp(loess=True, contours="vband", dwarfs=True,
                             letters=True):
    data = np.loadtxt(outtable, usecols=(69,72,75)).T
    xall, yall, sn = np.loadtxt("results.tab", usecols=(1,2,14,)).T
    ###############################################
    # Details of the maps
    names = [r"age", r"metal", r"alpha"]
    cb_label = [r"Age [Gyr]", r"[Z/H]", r"[$\alpha$/Fe]"]
    cmaps = [brewer2mpl.get_map('Reds', 'sequential', 9).mpl_colormap,
             brewer2mpl.get_map('Blues', 'sequential', 9).mpl_colormap,
             brewer2mpl.get_map('Greens', 'sequential', 9).mpl_colormap]
    # cmaps = [cm.rainbow] * 3
    ncor=9
    cmaps = [nc.cmap_discretize(nc.cmap_map(lambda x: x*0.5 + 0.45, x), ncor)
             for x in cmaps]
    xcb = [0.068, 0.385, 0.705]
    ###############################################
    # Set the threshold S/N for smoothing
    # Higher values than this values are not smoothed
    sn_thres = [25, 25, 25]
    ###############################################
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1,3)
    gs.update(left=0.051, right=0.985, bottom=0.10, top=0.98, hspace=0.06,
              wspace=0.06)
    cb_fmts=["%.1f","%.2f", "%.2f"]
    labels = ["(D)", "(E)", "(F)"]
    xi = np.linspace(-40, 40, 100)
    yi = np.linspace(-40, 40, 100)
    # Loop for figures
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(names[i])
        good = np.where(((~np.isnan(vector)) & (sn>sn_cut)))[0]
        if loess:
            sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_thres[i])))[0]
            sn_low = np.delete(good, sn_high)
            good = np.hstack((sn_high, sn_low ))
            vector_low = ll.loess_2d(xall[sn_low], yall[sn_low],
                                     vector[sn_low], frac=frac_loess)
            vector_high = vector[sn_high]
            v = np.hstack((vector_high, vector_low))
        else:
            v = vector[good]
        x = xall[good]
        y = yall[good]
        robust_sigma =  1.4826 * np.median(np.abs(v - np.median(v)))
        vmin = np.median(v) - 0.8 * robust_sigma
        vmax = np.median(v) + 0.8 * robust_sigma
        if i == 0:
            vmax = np.minimum(15., vmax)
        ylabels = [1,0,0]
        ax = plt.subplot(gs[i])
        norm = Normalize(vmin=vmin, vmax=vmax)
        coll = PolyCollection(polygons_bins[good], array=v, cmap=cmaps[i],
                                  edgecolors='none', norm=norm,
                                  linewidths=0.001)
        zi = griddata((x, y), v, (xi[None,:], yi[:,None]), method='linear')
        CS = plt.contourf(xi,yi,zi,3*ncor,cmap=cmaps[i], norm=norm)
        canvas.draw_slits(ax, amp=1.5)
        # draw_map(fig, ax, coll)
        ax.minorticks_on()
        lims=40
        ax.set_xlim([lims, -lims])
        ax.set_ylim([-lims, lims])
        draw_contours(contours, fig, ax)
        plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=10,
                            color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                      cbar_pos=[xcb[i], 0.16, 0.09, 0.04],
                      ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i])
        xylabels(ax, y=ylabels[i])
        if dwarfs:
            draw_galaxies(fig, ax)
        if i > 0:
            ax.set_yticklabels([])
        if letters:
            ax.text(0.05, 0.95, labels[i], transform=ax.transAxes,
                    fontsize=26, fontweight='bold', va='top')
    if loess:
        app = "loess"
    else:
        app = "noloess"
    plt.savefig("figs/populations_{0}_{1}.png".format(contours, app), dpi=120)
    return

def make_ssp(loess=False):
    """ Make single panel for 4 SSP parameters"""
    ########################################################
    # Read data values for Lick indices
    data = np.loadtxt(outtable, usecols=(69,72,75,84)).T
    # ads = np.loadtxt(outtable, usecols=(87,88,89,90)).T
    # Changing units of age for log scale
    # data[0] += 9.
    # Read spectra name
    s = np.genfromtxt(outtable, usecols=(0,), dtype=None).tolist()
    ########################################################
    # Read coords and S/N
    xall, yall, sn = np.loadtxt(outtable, usecols=(1,2, 14)).T
    ###############################################
    # Details of the maps

    cmaps = [brewer2mpl.get_map('Reds', 'sequential', 3).mpl_colormap,
             brewer2mpl.get_map('Blues', 'sequential', 3).mpl_colormap,
             brewer2mpl.get_map('Greens', 'sequential', 3).mpl_colormap,
             brewer2mpl.get_map('Oranges', 'sequential', 3).mpl_colormap]
    # cmaps = [nc.cmap_map(lambda x: x*0.54 + 0.43, x) for x in cmaps]
    cmaps = [nc.cmap_discretize(x, 6) for x in cmaps]
    titles = [r"age", r"total metallicity",
                r"element abundance", r"iron metallicity"]
    cb_label = [r"log Age (yr)", r"[Z/H]", r"[$\alpha$/Fe]", r"[Fe/H]"]
    lims = [[None, None], [None, None], [None, None], [None, None]]
    xcb = [0.105, 0.57]
    xcb = xcb + xcb
    yc1 = 0.565
    yc2 = 0.090
    ycb = [yc1, yc1, yc2, yc2]
    fig = plt.figure(figsize=(10.8, 10.2))
    gs = gridspec.GridSpec(2,2)
    gs.update(left=0.065, right=0.988, bottom=0.055, top=0.985, hspace=0.04,
              wspace=0.018)
    ylabels = [1,0,1,0]
    xlabels = [0,0,1,1]
    sn_thres = 25
    cb_fmts = ["%.2f", "%.2f", "%.2f", "%.2f"]
    ###############################################
    # Loop for figures
    for i, vector in enumerate(data):
        print "Producing figure for {0}...".format(titles[i])
        good = np.where(((np.isfinite(vector)) & (sn>sn_cut)))[0]
        # good = np.where(((ads[i] < 50.) & (np.isfinite(vector))))[0]
        v = vector[good]
        robust_sigma =  1.4826 * np.median(np.abs(v - np.median(v)))
        vmin = np.median(v)  - 0.8 * robust_sigma
        vmax = np.median(v) + 0.8 * robust_sigma
        vmin = lims[i][0] if lims[i][0] else vmin
        vmax = lims[i][1] if lims[i][1] else vmax
        if i == 0:
            vmax = np.minimum(np.log10(14.5 * 10**9), vmax)
        sn_high = np.where(((~np.isnan(vector)) & (sn>=sn_thres)))[0]
        sn_low = np.delete(good, sn_high)
        good = np.hstack((sn_high, sn_low ))
        if loess:
            vector_low = ll.loess_2d(xall[sn_low], yall[sn_low], vector[sn_low],
                                     frac=frac_loess)
            vector_high = vector[sn_high]
            v = np.hstack((vector_high, vector_low))
        else:
            v = vector[good]
        ax = plt.subplot(gs[i])
        norm = Normalize(vmin=vmin, vmax=vmax)
        coll = PolyCollection(polygons_bins[good], array=v, cmap=cmaps[i],
                              edgecolors='w', norm=norm, linewidths=1)
        draw_map(fig, ax, coll)
        draw_contours("vband", fig, ax)
        plt.gca().add_patch(Rectangle((18,-36),20,10, alpha=1, zorder=100000,
                            color="w"))
        draw_colorbar(fig, ax, coll, cblabel=cb_label[i],
                  cbar_pos=[xcb[i], ycb[i], 0.13, 0.02],
                  ticks=np.linspace(vmin, vmax, 4), cb_fmt=cb_fmts[i],
                  labelsize=30)
        xylabels(ax, y=ylabels[i], x=xlabels[i], size=16)
        if i not in [0,2]:
            ax.set_yticklabels([])
        if i < 2:
           ax.set_xticklabels([])
    nloess = "_loess" if loess else ""
    # plt.savefig("figs/ssps{0}.eps".format(nloess), dpi=1200, format="eps")
    plt.savefig("figs/ssps{0}.png".format(nloess), dpi=300)
    return

if __name__ == "__main__":
    plt.ioff()
    ####################################################
    # Set the fraction to be used in the smoothing maps
    frac_loess = 0.1
    ####################################################
    # Set the name of the table after merging tables
    ####################################################
    outtable = "results.tab"
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
    #######################################################
    # Use canvas properties to retrieve the id of the slits
    # Set slits to be used: 0-sky 1-halo 2-point sources, 3-HCC007
    #######################################################
    slits = [y for x,y in zip(canvas.slits.type, canvas.slits.ids) 
             if x in [1,3]]
    ignore_slits = ["cen2_s24", "inn2_s21", "cen1_s22",
                    "inn1_s22", "inn2_s27", "inn1s35"]
    slits = [x for x in slits if x not in ignore_slits]
    ####################################################
    # Positions of the slits
    xy = get_positions_by_slits(slits)
    ####################################################
    # Adding a hole in the position of NGC 3309
    hole = np.array([[-25., 9.]])
    ###############################################################
    # Create polygons
    ###############################################################
    polygons = make_voronoi(xy)
    ##############################################################
    # Merge polygons
    ##############################################################
    specs = np.loadtxt(outtable, usecols=(0,), dtype=str).tolist()
    if False:
        polygons_bins = merge_polys() 
    else:
        s = [x.split(".")[0].split("_", 1)[1][5:] for x in specs]
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
    # make_kinematics()
    # make_kin_summary(loess=1)
    ####################################################
    # Produce maps for Lick indices
    # make_lick2(loess=False, rlims=40)
    #Produce and array of maps
    # make_stellar_populations(loess=False, letters=0)
    # make_sp_panel(loess=False)
    # make_stellar_populations_horizontal()
    # make_ssp()
    #####################################################
    # make_other()
    # make_imgs()
    # make_sb(im="vband")
    #####################################################
    # Make interpolated maps with LOESS
    # make_maps_interp()

