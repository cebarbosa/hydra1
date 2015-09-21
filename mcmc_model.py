#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 11:35:13 2013

@author: cbarbosa
"""

import os
import shutil

import numpy as np
from scipy import stats
from scipy.interpolate import LinearNDInterpolator as interpolator
import pymc

from config import *

class SSP:
    """ Wrapper for the interpolated model."""
    def __init__(self, model_table, indices=np.arange(25)): 
        self.interpolate(model_table)
        self.indices = indices 
    
    def interpolate(self, model_table):
        modeldata = np.loadtxt(model_table, dtype=np.double)
        self.model = interpolator(modeldata[:,:3], modeldata[:,3:])
        
    def fn(self, age, metallicity, alpha): 
        return self.model(age, metallicity, alpha)[self.indices]
        
    def __call__(self, *args): 
        return self.fn(*args)

        
class Dist():
    """ Simple class to handle the distribution data of MCMC. """
    def __init__(self, dist, lims):
        self.data = dist
        self.pdf = stats.gaussian_kde(self.data)
        self.median = np.median(self.data)
        self.lims = lims
        self.x = np.linspace(lims[0], lims[1]+1, 1000)
        self.MAPP = self.x[np.argmax(self.pdf(self.x))]
        return

def get_coefficients(map_):
    return [{str(variable): variable.value} for variable in map_.variables]

def read_data(tab1, tab2):
    s1 = np.genfromtxt(tab1, usecols=(0,), dtype=None).tolist()
    s2 = np.genfromtxt(tab2, usecols=(0,), dtype=None).tolist()
    sref = [x for x in s1 if x in s2]
    sref.sort()
    data = np.loadtxt(tab1, usecols=np.arange(1,25))
    errs = np.loadtxt(tab2, usecols=np.arange(1,26))
    idx1 = np.array([s1.index(x) for x in sref])
    idx2 = np.array([s2.index(x) for x in sref])
    data = data[idx1]
    errs = errs[idx2]
    s1 = np.array(s1)[idx1]
    return s1, data, errs

def get_model_lims():
    """ Get the range for the indices according to models. """
    modeldata = np.loadtxt(os.path.join(tables_dir, "models_thomas_2010.dat"))
    indices = modeldata[:,3:].T
    lims = np.zeros((len(indices), 2))
    for i, index in enumerate(indices):
        lims[i] = [index.min(), index.max()]
    lim_excess =  0.5 * np.abs(np.diff(lims)).T[0]
    lims[:,0] -= lim_excess
    lims[:,1] += lim_excess
    return lims


if __name__ == "__main__":
    lims = [[0., 15.], [-2.25, 0.67], [-0.3, 0.5]]
    model_table = os.path.join(tables_dir, "models_thomas_2010.dat")
    age_dist = pymc.Uniform(name="age_dist", lower=1., upper=14.5)
    metal_dist = pymc.Uniform(name="metal_dist", lower=-2.25, upper=0.67)
    alpha_dist = pymc.Uniform(name="alpha_dist", lower=-0.3, upper=0.5)
    working_dir = os.path.join(home, "single2")
    os.chdir(working_dir)
    spectra, data, errs = read_data("lick_corr.tsv",
                                    "lick_mc_errs_10.txt")
    lims = get_model_lims()
    outtable = "ages_Z_alpha.tsv"
    for i, (spec, obsdata, obserr) in enumerate(zip(spectra, data, errs)):
        ######################################################################
        # Get valid indices
        nans = np.where(~np.isnan(obsdata))[0]
        indcols = np.where(~np.isnan(obsdata))[0]
        ######################################################################
        # Clip indices with unnexpected values according to model
        for idx in indcols:
             if obsdata[idx] <= lims[idx,0] or  obsdata[idx] > lims[idx,1]:
                 obsdata[idx] = np.nan
        ######################################################################
        dbname = spec.replace(".fits", "_db")
        dbfolder = os.path.join(working_dir, dbname)
        if os.path.exists(dbfolder):
            continue
        print "({0} / {1})".format(i+1, len(spectra))
        if i == 0:
            with open(outtable, "w") as f:
                f.write("# Spectra\tAge(Gyr)\tAge-\tAge+\t[Z/H]\t[Z/H]"
                        "-\t[Z/H]+\t[alpha/Fe]\t[alpha/Fe]-\t[alpha/Fe]+\n")
        csvfile = dbname + ".csv"
        if len(indcols) == 0:
            continue
        # Removing Mg1 and Mg2
        indcols = indcols[indcols != 14]
        indcols = indcols[indcols != 15]
        indcols = indcols[indcols>11] 
        obsdata = obsdata[indcols]
        obserr = obserr[indcols]
        taus = 1 / obserr**2
        ssp = SSP(model_table, indcols)
        @pymc.deterministic()
        def ssp1(age=age_dist, metal=metal_dist, alpha=alpha_dist):
            return ssp(age, metal, alpha)
        y = pymc.Normal(name="y", mu=ssp1, tau=taus,
                        value=obsdata, observed=True)

        model = pymc.Model([y, age_dist, metal_dist, alpha_dist]) 
        mcmc = pymc.MCMC(model, db="txt", dbname=dbname)
        mcmc.sample(10000, 1000, 3)
        mcmc.db.close()
        mcmc.summary()
        db = pymc.database.txt.load(dbname)
        mcmc = pymc.MCMC(model, db=db)
        ages = mcmc.trace("age_dist")[:]
        metals = mcmc.trace("metal_dist")[:]
        alphas = mcmc.trace("alpha_dist")[:]
        results = mcmc.stats(alpha=0.3173105)
        age_min, age_max = results["age_dist"]['68% HPD interval']
        metal_min, metal_max = results["metal_dist"]['68% HPD interval']
        alpha_min, alpha_max = results["alpha_dist"]['68% HPD interval']
        distage = Dist(ages, lims[0])
        distmetal = Dist(metals, lims[1])
        distalpha = Dist(alphas, lims[2])
        s = [distage.MAPP, age_min, age_max, distmetal.MAPP, metal_min, 
             metal_max, distalpha.MAPP, alpha_min, alpha_max]
#        pymc.raftery_lewis(mcmc, q=0.01, r=0.01)
        with open(outtable, "a") as f:
            f.write(spectra[i] + "\t" + 
                    "\t".join([str(round(x,5)) for x in s]) + "\n")
        break
