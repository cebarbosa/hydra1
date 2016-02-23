# -*- coding: utf-8 -*-
"""

Created on 22/02/16

@author: Carlos Eduardo Barbosa

Test for the importance of the flux calibration on the Lick indices

"""

import os

import numpy as np
import pyfits as pf
from scipy import ndimage
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from config import *
from ppxf import ppxf
import ppxf_util as util
from load_templates import stellar_templates
from run_ppxf import wavelength_array
import lector as lector
from mcmc_model import get_model_lims


def snr(flux, axis=0):
    """ Calculates the S/N ratio of a spectra.

    Translated from the IDL routine der_snr.pro """
    signal = np.nanmedian(flux, axis=axis)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.*flux - \
           np.roll(flux, 2, axis=axis) - np.roll(flux, -2, axis=axis)), \
           axis=axis)
    return signal / noise

def check_ppxf(spec, velscale):
    """ Checking if velocity os star is zero"""
    templates, logLam2, delta, miles= stellar_templates(velscale)
    FWHM_dif = np.sqrt(FWHM_tem**2 - FWHM_spec**2)
    sigma = FWHM_dif/2.355/delta # Sigma difference in pixels
    star = pf.getdata(spec)
    star1 = ndimage.gaussian_filter1d(star, sigma)
    w = wavelength_array(spec)
    lamRange= np.array([w[0], w[-1]])
    galaxy, logLam1, velscale = util.log_rebin(lamRange, star1, \
                                               velscale=velscale)
    sn = snr(star1)
    noise = np.ones_like(galaxy) / sn
    dv = (logLam2[0]-logLam1[0])*c
    pp = ppxf(templates, galaxy, noise, velscale, [0,50],
        plot=True, moments=2, degree=5, mdegree=-1, vsyst=dv)
    plt.show()
    return

def data_schiavon07():
    """ Lick indices from the Schiavon et al. 2007. """
    return np.array([-1.023, -4.059, 0.187, 0.214, 0.400, 6.237, -1.826,
                     -7.399, 4.756, np.nan, np.nan, 6.364, 1.897, 5.994,
                     np.nan, 0.131, 2.110, 2.979, 2.490, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan])

def test_star():
    os.chdir(os.path.join(home, "data/star"))
    specs = ['HD102070_spectra.fits', "HD102070_noflux.fits"]
    a = np.zeros((2,25))
    cs = ["k", "r"]
    fs = []
    for i, spec in enumerate(specs):
        star = pf.getdata(spec)
        w = wavelength_array(spec)
        # plt.plot(w, star/np.nanmedian(star), "-{0}".format(cs[i]))
        # check_ppxf(spec, velscale) # Velocity of stars is zero
        flux = lector.broad2lick(w, star, 2.1, vel=0.)
        sn = snr(flux)
        noise = np.ones_like(flux) * np.median(flux) / sn
        lick, lickerrs = lector.lector(w, flux, noise, bands,
                 vel = 0., cols=(0,8,2,3,4,5,6,7))
        a[i] = lick
        fs.append(interp1d(w, star/np.nanmedian(star), bounds_error=False, fill_value=0))
    # plt.show()
    # plt.clf()
    w = np.linspace(4850, 5780, 1000)
    p = np.poly1d(np.polyfit(w, fs[0](w) / fs[1](w), 20))
    # plt.plot(w, fs[0](w) / fs[1](w), "-k" )
    # plt.plot(w, p(w), "-r")
    # plt.show()
    cols = ["Index       ", "Calib", "Raw", "Sch07", "Delta", "%", "offset"]
    cols = ["{0:10s}".format(x) for x in cols]
    sch = data_schiavon07()
    model_table = os.path.join(tables_dir, \
                               "models_thomas_2010_metal_extrapolated.dat")
    lims, ranges = get_model_lims(model_table, factor=0)
    lims = np.diff(lims).T[0]
    print "Results for the test on standard star HD 102070"
    print "".join(cols)
    for j,index in enumerate(indices):
        if j < 12 or j>20:
            continue
        print "{0:6s}{1:10.2f}{2:10.2f}{6:10.2f}{3:10.2f}{4:10.2f}{5:10.2f}".format(index, \
                a[0][j],  a[1][j], (a[0][j] - a[1][j]),
                (a[0][j] - a[1][j])/ a[0][j], offset[j], sch[j])
    return w,p

def test_galaxy(wpoly, poly):
    os.chdir(os.path.join(home, "single1"))
    specs = ["fin1_n3311cen1_s27.fits", "fin1_n3311cen2_s30.fits",
             "fin1_n3311inn1_s28.fits", "fin1_n3311inn2_s28.fits"]
    kfile = "ppxf_results.dat"
    kspecs = np.loadtxt(kfile, usecols=(0,), dtype=str)
    v = np.loadtxt(kfile, usecols=(1,))
    vs = dict([(x,y) for x,y in zip(kspecs, v)])
    araw = np.zeros((len(specs), 25))
    aflux = np.zeros_like(araw)
    for i,spec in enumerate(specs):
        galaxy = pf.getdata(spec)
        w = wavelength_array(spec)
        flux = lector.broad2lick(w, galaxy, 2.1, vel=vs[spec])
        sn = snr(flux)
        noise = np.ones_like(flux) * np.median(flux) / sn
        lick, lickerrs = lector.lector(w, flux, noise, bands,
                 vel = vs[spec], cols=(0,8,2,3,4,5,6,7))
        araw[i] = lick
        galaxy2 = galaxy * poly(w)
        flux2 = lector.broad2lick(w, galaxy2, 2.1, vel=0.)
        lick2, lickerrs2 = lector.lector(w, flux2, noise, bands,
                 vel = vs[spec], cols=(0,8,2,3,4,5,6,7))
        aflux[i] = lick2
    araw = araw.T
    aflux = aflux.T
    for j,index in enumerate(indices):
        if j < 12 or j>20:
            continue
        print indices[j],
        # print aflux[j],
        # print araw[j],
        # print lims[j]
        print "{0:10.2f}".format(np.median(aflux[j])),
        print "{0:10.2f}".format(np.median(araw[j])),
        print "{0:10.2f}".format(np.median((aflux[j] - araw[j])/(aflux[j])))

        # print "{0:6s}{1:10.2f}{2:10.2f}{6:10.2f}{3:10.2f}{4:10.2f}{5:10.2f}".format(index, \
        #         a[0][j],  a[1][j], (a[0][j] - a[1][j]),
        #         (a[0][j] - a[1][j])/ a[0][j], offset[j], sch[j])




if __name__ == "__main__":
    bands = os.path.join(tables_dir, "BANDS")
    offset = np.loadtxt(os.path.join(tables_dir,"LICK_OFFSETS.dat"),
                        usecols=(1,)).T
    indices = np.loadtxt(bands, usecols=(0,), dtype=str)
    model_table = os.path.join(tables_dir, \
                               "models_thomas_2010_metal_extrapolated.dat")
    lims, ranges = get_model_lims(model_table, factor=0)
    w,p = test_star()
    test_galaxy(w,p)



