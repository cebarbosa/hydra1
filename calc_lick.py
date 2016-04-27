# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:01:07 2013

@author: cbarbosa

Study velocity dispersion correction of Lick indices in the Hydra I sample
"""

import os
import pickle

import numpy as np
import pyfits as pf
from scipy.interpolate import NearestNDInterpolator as interpolator
from scipy.interpolate import interp1d

from config import *
import lector as lector
from run_ppxf import pPXF

def correct_indices(indices, inderr, indtempl, indtempl_b, types):
    """ Make corrections for the broadening in the spectra."""
    corrected = np.zeros_like(indices)
    errors = np.zeros_like(indices)
    for i,t in enumerate(types):
        if t == 0:
            C = indtempl[i] / indtempl_b[i]
            if C >= 1:
                corrected[i] = indices[i] * C
                errors[i] = C * inderr[i]
            else:
                corrected[i] = indices[i] 
                errors[i] = inderr[i]
#                print "Warning: Meaningless correction for {0}.".format(
#                       lick_indices[i])
        elif t in [1, 2]:
            C = indtempl[i] - indtempl_b[i]
            corrected[i] = indices[i] + C
            errors[i] = inderr[i]
#        print lick_indices[i], t, indices[i], C, corrected[i]
    return corrected, errors
    
def correct_indices2(indices, inderr, indtempl, indtempl_b, types):
    """ Make corrections for the broadening in the spectra."""
    corrected = np.zeros_like(indices)
    errors = np.zeros_like(indices)
    for i,t in enumerate(types):
        C = indtempl[i] - indtempl_b[i]
        corrected[i] = indices[i] + C
        errors[i] = inderr[i]
    return corrected, errors
    
def save(results, output):
    """ Save results to a output file. """
    with open(output, "w") as f:
        f.write("\n".join(results))
    
def wavelength_array(spec):
    """ Produces array for wavelenght of a given array. """
    w0 = pf.getval(spec, "CRVAL1")
    deltaw = pf.getval(spec, "CD1_1")
    pix0 = pf.getval(spec, "CRPIX1")
    npix = pf.getval(spec, "NAXIS1")
    return w0 + deltaw * (np.arange(npix) + 1 - pix0)    

def check_intervals(setupfile, bands, vel):
    """ Check which indices are defined in the spectrum. """
    c = 299792.458 # speed of light in km/s
    with open(setupfile) as f:
        lines = [x for x in f.readlines()]
    lines = [x for x in lines if x.strip()]
    intervals = np.array(lines[5:]).astype(float)
    intervals = intervals.reshape((len(intervals)/2, 2))
    bands = np.loadtxt(bands, usecols=(2,7))
    bands *= np.sqrt((1 + vel/c)/(1 - vel/c))
    goodbands = np.zeros(len(bands))
    for i, (b1, b2) in enumerate(bands):
        for (i1, i2) in intervals:
            if i1 < b1 and b2 < i2:
                goodbands[i] = 1
    return np.where(goodbands == 1, 1, np.nan)

class BroadCorr:
    """ Wrapper for the interpolated model."""
    def __init__(self, table): 
        self.interpolate(table)
    
    def interpolate(self, table):
        sigmas = np.loadtxt(table, dtype=np.double, usecols=(0,))
        inds = np.loadtxt(table, dtype=np.double, usecols=np.arange(1,51, 2)).T
        corrs = np.loadtxt(table, dtype=np.double,usecols=np.arange(2,51, 2)).T
        self.fs = []
        self.indlims = []
        for i, (idx, corr) in enumerate(zip(inds, corrs)):
            f = interpolator(np.column_stack((sigmas, idx)), corr)
            self.fs.append(f)
            self.indlims.append([idx.min(), idx.max()])
        self.siglims = [sigmas.min(), sigmas.max()]
        return
        
    def __call__(self, sigma, lick): 
        b = np.zeros(len(lick))
        for i,l in enumerate(lick):
            if np.isnan(l):
                b[i] = 0.
            else:
                b[i] = self.fs[i](sigma, l)
        return b

class Vdisp_corr_k04():
    """ Correction for LOSVD only for multiplicative indices from
        Kuntschner 2004."""
    def __init__(self):
        """ Load tables. """
        table = os.path.join(tables_dir, "kuntschner2004.tab")
        bands = os.path.join(tables_dir, "BANDS")
        self.indices_k04 = np.loadtxt(table, usecols=(1,), dtype=str).tolist()
        self.type_k04 = np.loadtxt(table, usecols=(2,),
                                            dtype=str).tolist()
        self.coeff_k04 = np.loadtxt(table, usecols=np.arange(3,10))
        self.lick_indices = np.loadtxt(bands, usecols=(0,), dtype=str)
        self.lick_types = np.loadtxt(bands, usecols=(8,))

    def __call__(self, lick, sigma, h3=0., h4=0.):
        newlick = np.zeros(25)
        for i,index in enumerate(lick_indices):
            if index in self.indices_k04:
                idx = self.indices_k04.index(index)
                a1, a2, a3, b1, b2, c1, c2 = self.coeff_k04[idx]
                if self.type_k04[idx] == "m":
                    C_k04 = 1. + a1 * sigma + a2 * sigma**2 + \
                            a3 * sigma**3 + b1 * sigma * h3 + \
                            b2 * sigma**2 * h3 + \
                            c1 * sigma * h4 + c2 * sigma**2 * h4
                    newlick[i] = C_k04 * lick[i]
                else:
                    C_k04 = a1 * sigma    + a2 * sigma**2 + \
                            a3 * sigma**3 + b1 * sigma * h3 + \
                            b2 * sigma**2 * h3 + \
                            c1 * sigma * h4 + c2 * sigma**2 * h4
                    newlick[i] = lick[i] + C_k04
            else:
                newlick[i] = lick[i]
        return newlick

if __name__ == "__main__":
    workdir = os.path.join(home, "single2")
    os.chdir(workdir)
    if not os.path.exists(os.path.join(workdir, "logs")):
        os.mkdir(os.path.join(workdir, "logs"))
    kinfile = "ppxf_results.dat"
    specs = np.genfromtxt(kinfile, usecols = (0,), dtype=None).tolist()
    K04 = Vdisp_corr_k04()
    bands = os.path.join(tables_dir, "BANDS")
    lick_types = np.loadtxt(bands, usecols=(8,))
    lick_indices = np.genfromtxt(bands, usecols=(0,), dtype=None).tolist()
    header = "# Spectra\t" + "\t".join(lick_indices)
    # Initiating outputss
    results, results_err = [header], [header]
    results5, results5_err = [header], [header]
    bcorr = BroadCorr(os.path.join(tables_dir, "lickcorr_m.txt"))
    offset = np.loadtxt(os.path.join(tables_dir,"LICK_OFFSETS.dat"),
                        usecols=(1,)).T
    broad2lick = True
    for i, spec in enumerate(specs):
        setupfile = os.path.join(home, "single1/{0}.setup".format(spec))
        if not os.path.exists(setupfile):
            print "Setup file not found: ", setupfile
            continue
        # Read the spectrum file and pPXF results
        pp = pPXF(spec, velscale)
        pp.calc_arrays_emission()
        pp.sky_sub()
        if pp.ncomp > 1:
            v, s, h3, h4 = pp.sol[0]
        else:
            v, s, h3, h4 = pp.sol
        print spec, v, s
        goodindices = check_intervals(setupfile, bands, v)
        ######################################################################
        # Check problem with broadening
        bf = interp1d(pp.w_log, pp.bestfit, bounds_error=False,
                      fill_value="extrapolate")
        bestfit = bf(pp.w)
        bfu = interp1d(pp.w_log, pp.bestfit_unbroad, bounds_error=False,
                      fill_value="extrapolate")
        bestfitunb = bfu(pp.w)
        ######################################################################
        # Broadening of the spectra to the Lick resolution
        if broad2lick:
            pp.flux = lector.broad2lick(pp.w, pp.flux, 2.1, vel=v)
            pp.bestfit = lector.broad2lick(pp.w_log, pp.bestfit, 2.54, vel=v)
            pp.bestfit_unbroad = lector.broad2lick(pp.w_log, pp.bestfit_unbroad,
                                                   2.54, vel=v)
            flux = lector.broad2lick2(pp.w, pp.flux, 2.1, vel=v)
            bestfit = lector.broad2lick2(pp.w, bestfit, 2.54, vel=v)
            bestfitunb = lector.broad2lick2(pp.w, bestfitunb, 2.54, vel=v)
        noise = pp.flux / pp.noise[0]
        #####################################################################
        # Make Lick indices measurements
        #####################################################################
        lick, lickerrs = lector.lector(pp.w, pp.flux-pp.em_linear, noise, bands,
                         vel = v, cols=(0,8,2,3,4,5,6,7),
                         keeplog=0, output="logs/lick_{0}".format(
                         spec.replace(".fits", ".pdf")), title=spec)
        # Measuring Lick indices in the templates
        noise2 = pp.bestfit / pp.noise[0]
        lick_bf, tmp = lector.lector(pp.w_log, pp.bestfit - pp.em, noise2, bands,
                         vel = v, cols=(0,8,2,3,4,5,6,7),
                         keeplog=0, output="a1.pdf", title=spec)
        noise3 = pp.bestfit_unbroad / pp.noise[0]
        lick_bf_unb, tmp = lector.lector(pp.w_log, pp.bestfit_unbroad - pp.em,
                                         noise3,
                                        bands, vel = v, cols=(0,8,2,3,4,5,6,7),
                                        keeplog=0, title=spec)
        ####################################################################
        # Measure in new specs
        noise2 = pp.bestfit / pp.noise[0]
        lnew, tmp = lector.lector(pp.w, flux-pp.em_linear, noise, bands,
                         vel = v, cols=(0,8,2,3,4,5,6,7),
                         keeplog=0, output="logs/lick_{0}".format(
                         spec.replace(".fits", ".pdf")), title=spec)
        lbf, tmp = lector.lector(pp.w, bestfit-pp.em_linear, noise, bands,
                         vel = v, cols=(0,8,2,3,4,5,6,7),
                         keeplog=0, output="a1.pdf", title=spec)
        noise3 = pp.bestfit_unbroad / pp.noise[0]
        lbfu, tmp = lector.lector(pp.w, bestfitunb-pp.em_linear, noise,
                                        bands, vel = v, cols=(0,8,2,3,4,5,6,7),
                                        keeplog=0, title=spec)
        ####################################################################
        igood = np.array([12,13,16,17,18,19,20])
        # Removing bad indices
        lick *= goodindices
        lickerrs *= goodindices
        lick_bf *= goodindices
        lick_bf_unb *= goodindices
        ####################################################################
        # LOSVD correction using averages over templates
        lick2 = lick * bcorr(s, lick)
        lickerrs2 = lickerrs * bcorr(s, lick)
        ####################################################################
        # LOSVD correction using best fit templates
        ####################################################################
        lick3, lickerrs3 = correct_indices(lick, lickerrs, lick_bf_unb, lick_bf,
                                         lick_types)
        ######################################################################
        # LOSVD correction for some indices using Kuntschner 2004.
        ######################################################################
        lick4 = K04(lick, s, h3, h4)
        ######################################################################
        # New correction
        lick5, lickerrs5 = correct_indices(lnew, lickerrs, lbfu, lbf,
                                         lick_types)
        ######################################################################
        # Offset correction
        lick += offset
        lick2 += offset
        lick3 += offset
        lick4 += offset
        lick5 += offset
        ######################################################################
        # Convert to string
        ######################################################################
        lick = "".join(["{0:14}".format("{0:.5f}".format(x)) for x in lick])
        lick2 = "".join(["{0:14}".format("{0:.5f}".format(x)) for x in lick2])
        lick5 = "".join(["{0:14}".format("{0:.5f}".format(x)) for x in lick5])
        # Append to output
        results.append("{0:28s}".format(spec) + lick)
        results5.append("{0:28s}".format(spec) + lick5)
    res = "lickres" if broad2lick else "instres"
    save(results5, "lick_vdcorr_{0}.tsv".format(res))
    save(results, "lick_novdcorr_{0}.tsv".format(res))