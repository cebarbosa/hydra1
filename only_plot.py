# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 01:09:38 2014

@author: kadu

Make plot of examples of pPFX fitting. Equivalent of only_plot_dat in Lodo's
IDL program, but much slower.
"""
import os 
import pickle

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

from config import *
from run_ppxf import pPXF
import ppxf_util as util

def get_ranges(spec):
    filename = os.path.join(data_dir,spec + ".setup")
    with open(filename) as f:
        f.readline()
        start = f.readline().split()
    ranges = np.loadtxt(filename, skiprows=5)
    return np.reshape(ranges, (-1, 2))
    
def w_temp(velscale):
    """ Make templates array"""
    current_dir = os.getcwd()
    os.chdir(template_dir)
    miles = [x for x in os.listdir(".") if x.endswith(".fits")]
    miles.sort()
    c = 299792.458
    FWHM_tem = 2.54 # MILES library spectra have a resolution FWHM of 2.54A.
    # Extract the wavelength range and logarithmically rebin one spectrum
    # to the same velocity scale of the SAURON galaxy spectrum, to determine
    # the size needed for the array which will contain the template spectra.
    #
    hdu = pf.open(miles[0])
    ssp = hdu[0].data
    h2 = hdu[0].header
    lamRange2 = h2['CRVAL1'] + np.array([0.,h2['CDELT1']*(h2['NAXIS1']-1)])
    sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, 
                                               velscale=velscale)
    os.chdir(current_dir)
    return np.exp(logLam2)

def get_lick_regions():
    """ Read definitions of bands of the Lick indices and return bands. """
    table = os.path.join(tables_dir, "BANDS")
    return np.loadtxt(table, usecols=(2,3)), np.loadtxt(table, usecols=(4,5)),\
           np.loadtxt(table, usecols=(6,7))

if __name__ == "__main__":
    wdir = home + "/single2"
    os.chdir(wdir)
    # plt.switch_backend('macosx')
    plt.ioff()
    outfile = PdfPages("ppxf_results.pdf")
    specs= np.loadtxt("ppxf_results.dat", dtype=str, usecols=(0,)).tolist()
    # Workaround to deal with cases where you have only one object in the file
    if isinstance(specs, str):
        specs = [specs]
    ###########################################################################
    xlims = [4800, 5850]
    w_tem = w_temp(velscale)
    red, bands, blue = get_lick_regions()
    names = [r'Hd$_A$', r'Hd$_F$', r'CN$_1$', r'CN$_2$', 'Ca4227',
             r'G4300', r'Hg_A', r'Hg_F', r'Fe4383', r'Ca4455',
             r'Fe4531', r'C4668', r'H$\beta$', r'Fe5015', r'Mg$_1$',
             r'Mg$_2', r'Mg $b$', r'Fe5270', r'Fe5335', r'Fe5406', r'Fe5709',
             r'Fe5782', r'Na_D', r'TiO_1', r'TiO_2']
    textsize = 16
    for i, spec in enumerate(specs):
        print spec,
        name = spec.replace(".fits", '').replace("n3311", "").split("_")
        name = name[1] + name[2]
        name = r"{0}".format(name)
        plt.minorticks_on()
        pp = pPXF(spec, velscale, pklfile=spec.replace(".fits", ".pkl"))
        pp.calc_sn()
        if pp.ncomp > 1:
            sol = pp.sol[0]
            error = pp.error[0]
            sol2 = pp.sol[1]
            error2 = pp.error[1]
        else:
            sol = pp.sol
            error = pp.error
        # Emission line
        em_weights = pp.weights[-3:]
        em_matrix = pp.matrix[:,-3:]
        em = em_matrix.dot(em_weights)
        f = interp1d(pp.w_log, em, kind="linear", bounds_error=False,
                     fill_value=0. )
        em_lin = f(pp.w)
        ######################################################################
        plt.plot(pp.w_log[pp.goodpixels], pp.galaxy[pp.goodpixels], "-k")
        plt.plot(pp.w_log[pp.goodpixels], pp.bestfit[pp.goodpixels], "-r",
                 lw=1.5)
        plt.plot(pp.w_log[pp.goodpixels],
                 pp.bestfit[pp.goodpixels] - em[pp.goodpixels], "--y")
        plt.plot(pp.w_log[pp.goodpixels], em[pp.goodpixels], "-b", lw=1.5)
        diff = pp.galaxy[pp.goodpixels] - pp.bestfit[pp.goodpixels]
        plt.plot(pp.w_log[pp.goodpixels], diff, ".g", ms=0.5)
        badpixels = np.setdiff1d(np.arange(len((pp.w_log))), pp.goodpixels)
        badpixels.sort()  
        ymin = np.floor(np.min(diff))
        ymax = 1.2 * plt.ylim()[1]
        plt.xlim(xlims[0], xlims[1])
        ylim = plt.ylim(ymin, ymax)
        plt.plot(pp.w_log[badpixels], 
                 pp.flux_log[badpixels] - pp.bestfit[badpixels], 
                 ".k", ms=0.5)
        plt.ylim(ylim)         
        plt.axhline(y=0, ls="--", c="k")
        plt.xlabel(r"$\lambda$ ($\AA$)", size=18)
        plt.ylabel(r"Flux (Counts)", size=18)
        plt.tight_layout()
        plt.annotate("{0}".format(name.upper()), xycoords='axes fraction',
                    xy=(0.05,0.95), size=textsize)
        plt.annotate(r"$\chi^2=${0:.2f}".format(pp.chi2), xycoords='axes fraction',
                    xy=(0.05,0.88), size=textsize)
        plt.annotate(r"S/N={0}".format(np.around(pp.sn,1)),
                     xycoords='axes fraction', xy=(0.25,0.95), size=textsize)
        plt.annotate(r"V={0} km/s".format(np.around(sol[0])),
                     xycoords='axes fraction', xy=(0.45,0.95), size=textsize,
                     color="r")
        plt.annotate(r"$\sigma$={0} km/s".format(np.around(sol[1])),
                     xycoords='axes fraction', xy=(0.75,0.95), size=textsize,
                     color="r")
        if pp.ncomp > 1:
            plt.annotate(r"V={0} km/s".format(np.around(sol2[0])),
                         xycoords='axes fraction', xy=(0.45,0.88),
                         size=textsize, color="b")
            plt.annotate(r"$\sigma$={0} km/s".format(np.around(sol2[1])),
                         xycoords='axes fraction', xy=(0.75,0.88),
                         size=textsize, color="b")
        y0, y1 = plt.ylim()
        bands_shift = bands * np.sqrt((1 + sol[0]/c)/(1 - sol[0]/c))
        for j, (lamb1, lamb2) in enumerate(bands_shift):
            if j in [14,15]:
                continue
            plt.fill_between([lamb1, lamb2], [y0, y0], [y1, y1], color="0.7")
            plt.annotate(names[j],
                     xycoords='data',
                     xy=(np.average(bands_shift[j]-10), 0.2 * (y0 + y1)),
                     rotation=90)
        bands_shift = red * np.sqrt((1 + sol[0]/c)/(1 - sol[0]/c))
        for j, (lamb1, lamb2) in enumerate(bands_shift):
            if j in [14,15]:
                continue
            plt.fill_between([lamb1, lamb2], [y0, y0], [y1, y1], color="0.9")
        bands_shift = blue * np.sqrt((1 + sol[0]/c)/(1 - sol[0]/c))
        for j, (lamb1, lamb2) in enumerate(bands_shift):
            if j in [14,15]:
                continue
            plt.fill_between([lamb1, lamb2], [y0, y0], [y1, y1], color="0.9")
        plt.savefig("logs/ppxf_{0}.png".format(name), dpi=100)
        plt.pause(0.001)
        # raw_input()
        outfile.savefig()
        plt.clf()
    outfile.close()