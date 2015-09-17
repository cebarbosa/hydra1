#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 27/01/15 19:42

@author: Carlos Eduardo Barbosa

Program to obtain indices from Loubser + 2012 and the stellar population
parameters

"""
import os
import shutil

import numpy as np
import pymc
import scipy.ndimage as ndimage

from config import *
from mcmc_model import SSP
from mcmc_analysis import Dist
import canvas as cv
from maps import polar2cart

def calc_indices():
    """ Calculate the indices in Angstrons for data of Loubser+ 2012"""
    filenames = ["hbeta.csv", "mgb.csv", "fe5270.csv", "fe5335.csv"]
    delta = [4876.625 - 4847.875, 5192.625 - 5160.125,
             5285.650 - 5245.650, 5352.125 - 5312.125]
    err_mag = np.array([0.025, 0.018, 0.02, 0.022])
    results = []
    radius = []
    err = []
    for i, filename in enumerate(filenames):
        data = np.loadtxt(filename, skiprows=1, delimiter=",")
        data[:,0] /= 10
        data[:,1] /= 100
        indices = delta[i] * (np.power(10, 0.4 * data[:,1]) - 1)
        radius.append(data[:,0])
        results.append(indices)
        err.append(delta[i] * np.power(10, 0.4 * data[:,1]) * np.log(10) * 0.4 * err_mag[i])
    err = np.array(err).T
    radius = np.array(radius)
    radius = radius.mean(axis=0)
    results = np.array(results).T
    a = np.column_stack((radius, results))
    np.savetxt(os.path.join(tables_dir, "loubser2012_scanned.dat"), a)
    return radius, results, err

def calc_sb(rlog, im="vband"):
    """ Calculate the surface brightness in the V-band """
    ###############################################
    # Set the background images for contours
    # Read values of S/N
    if im == "vband":
        canvas = cv.CanvasImage("vband")
    else:
        canvas_res = cv.CanvasImage("residual")
    datasmooth = ndimage.gaussian_filter(canvas.data, 30, order=0.)
    r = np.power(10, rlog) * 26.6
    pa = 63. * np.ones_like(r)
    x, y = polar2cart(r, pa)
    x0, x1, y0, y1 = canvas.extent
    ysize, xsize = canvas.data.shape
    ypix =  (y-y0)/(y1-y0) * ysize
    xpix = (x-x0)/(x1-x0) * xsize
    sb = []
    for xx,yy in zip(xpix, ypix):
        sb.append(datasmooth[yy,xx])
    return np.array(sb)

def read_tables():
    """ Read tables Ilani Loubser sent to me in private communication. """
    data = np.zeros((7, 25))
    errors = np.zeros_like(data)
    for i in range(7):
        table = 'NGC3311.000{0}.txt'.format(i+1)
        data[i] = np.loadtxt(table, usecols=(1,), skiprows=1)
        errors[i] = np.loadtxt(table, usecols=(6,), skiprows=1)

    return data, errors

def lick_mag_to_ang(data, doerr=False, errs=None):
    """ Convert all Lick indices to Angstrons"""
    licktable = os.path.join(tables_dir, "BANDS")
    names = np.loadtxt(licktable, usecols=(0,), dtype=str)
    types = np.loadtxt(licktable, usecols=(8,))
    deltal = np.diff(np.loadtxt(licktable, usecols=(4,5)))[:,0].T
    newdata = np.copy(data)
    if doerr:
        newerr = np.copy(errs)
    ii = np.where(types==1)[0]
    for j in range(len(data)):
        for i in ii:
            if doerr:
                newerr[j,i] = np.abs(deltal[i] * np.power(10, 0.4 * data[j,i]) *
                                    np.log(10) * 0.4 * errs[j,i])
            newdata[j,i] = (np.power(10, 0.4 * data[j,i]) - 1) * deltal[i]
    if doerr:
        return newdata, newerr
    return newdata

def convert_model():
    """ Converting the model of Thomas+ 2010 to angstrons. """
    model_table = os.path.join(tables_dir, "models_thomas_2010.dat")
    model_pars = np.loadtxt(model_table, usecols=(0,1,2))
    model_data = np.loadtxt(model_table, usecols=np.arange(3,28))
    newmodel = lick_mag_to_ang(model_data)

    newmodel = np.column_stack((model_pars, newmodel))
    newmodel_file = os.path.join(tables_dir,
                           "models_thomas_2010_angstroms.dat")
    np.savetxt(newmodel_file, newmodel)
    return

def calc_populations(data, errs, overwrite=0):
    model_table = os.path.join(tables_dir, "models_thomas_2010_angstroms.dat")
    lims = [[0., 15.], [-2.25, 0.67], [-0.3, 0.5]]
    pops = []
    for i in range(len(data)):
        age_dist = pymc.Uniform(name="age_dist", lower=1., upper=14.5)
        metal_dist = pymc.Uniform(name="metal_dist", lower=-2.25, upper=0.67)
        alpha_dist = pymc.Uniform(name="alpha_dist", lower=-0.3, upper=0.5)
        indcols = np.arange(12, 19)
        indcols = indcols[indcols != 14]
        indcols = indcols[indcols != 15]
        ssp = SSP(model_table, indcols)
        taus = 1 / errs[i]**2
        obsdata = data[i]
        obsdata = obsdata[indcols]
        taus = taus[indcols]

        @pymc.deterministic()
        def ssp1(age=age_dist, metal=metal_dist, alpha=alpha_dist):
            return ssp(age, metal, alpha)
        y = pymc.Normal(name="y", mu=ssp1, tau=taus,
                        value=obsdata, observed=True)
        dbname = "r{0:.2f}.db".format(radius[i])
        model = pymc.Model([y, age_dist, metal_dist, alpha_dist])
        if overwrite:
            shutil.rmtree(dbname)
        if not os.path.exists(dbname):
            mcmc = pymc.MCMC(model, db="txt", dbname=dbname)
            mcmc.sample(10000, 1000, 3)
            mcmc.db.close()
            mcmc.summary()
        db = pymc.database.txt.load(dbname)
        mcmc = pymc.MCMC(model, db=db)
        ages = mcmc.trace("age_dist")[:]
        metals = mcmc.trace("metal_dist")[:]
        alphas = mcmc.trace("alpha_dist")[:]
        mcmc_results = mcmc.stats(alpha=0.3173105)
        age_min, age_max = mcmc_results["age_dist"]['68% HPD interval']
        metal_min, metal_max = mcmc_results["metal_dist"]['68% HPD interval']
        alpha_min, alpha_max = mcmc_results["alpha_dist"]['68% HPD interval']
        distage = Dist(ages, lims[0])
        distmetal = Dist(metals, lims[1])
        distalpha = Dist(alphas, lims[2])
        s = [distage.MAPP, age_min, age_max, distmetal.MAPP, metal_min,
             metal_max, distalpha.MAPP, alpha_min, alpha_max]
        pops.append(s)
    pops = np.column_stack((radius, np.array(pops), sb))
    with open(os.path.join(tables_dir, "loubser12_populations.dat"), "w") as f:
        f.write("# Log radius (effective radius)\tAge(Gyr)\tAge-\tAge+\t[Z/H]\t[Z/H]"
                "-\t[Z/H]+\t[alpha/Fe]\t[alpha/Fe]-\t[alpha/Fe]+\tV-band SB\n")
        np.savetxt(f, pops)
    return

if __name__ == "__main__":
    os.chdir(home + "/loubser2012")
    radius, results, err = calc_indices()
    # indices, errs = read_tables()
    sb = np.loadtxt(os.path.join(tables_dir, "loubser12_populations.dat"),
                                 usecols=(10,))
    # sb = calc_sb(radius)
    # indices, errs = lick_mag_to_ang(indices, errs=errs, doerr=1)
    # with open("lick_loubser2012.txt", "w") as f:
    #     np.savetxt(f, np.column_stack((radius, indices)))
    # with open("lick_loubser2012_errs.txt", "w") as f:
    #     np.savetxt(f, np.column_stack((radius, errs)))
    # weights = (indices / errs)**2
    # weights /= weights.max()
    # np.set_printoptions(3, suppress=1)
    # print weights.mean(axis=0)
    # raw_input()
    ##########################################################################
    # Converting model of Thomas+ 2010
    # convert_model()
    # calc_populations(indices, errs, overwrite=1)
    #########################################################################
    # Analysis of Chains
    results = []
    for r,s in zip(np.around(radius, 2), sb):
        db = r"r{0}.db".format(r)
        ages_data = np.loadtxt("{0}/Chain_0/age_dist.txt".format(db))
        ages_data = np.log10(ages_data)
        ages = Dist(ages_data, [np.log10(1),np.log10(15)])
        metal_data = np.loadtxt("{0}/Chain_0/metal_dist.txt".format(db))
        metal = Dist(metal_data, [-2.25, 0.67])
        alpha_data = np.loadtxt("{0}/Chain_0/alpha_dist.txt".format(db))
        alpha = Dist(alpha_data, [-0.3, 0.5])
        line = [r, ages.MAPP, ages.MAPPmin, ages.MAPPmax, metal.MAPP,
                metal.MAPPmin, metal.MAPPmax, alpha.MAPP, alpha.MAPPmin,
                alpha.MAPPmax, s]
        results.append(line)
    results = np.array(results)
    np.savetxt(os.path.join(tables_dir, "loubser12_populations.txt"), results)








