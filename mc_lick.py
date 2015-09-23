# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:22:29 2014

@author: cbarbosa

Make simulations to derive errors on Lick indices including both noise and the 
error on the determined velocity.
"""
import numpy as np
import multiprocessing as mp
from scipy.stats import nanmedian

from setup_n3311 import *
import lector
from calc_lick import BroadCorr
from run_ppxf import pPXF, speclist

def run_mc(spec, i):
    """ Run MC routine in single spectrum. """
    outname = "mc_logs/{0}_nsim{1}.txt".format(spec.replace(".fits", ""),
                                               Nsim)
    output = os.path.join(wdir, outname)
    if os.path.exists(output):
        return
    print "{0} ({1}/{2})".format(spec, i+1, len(specs))
    pp = pPXF(spec, velscale)
    sn = pp.calc_sn()
    #####################################################################
    # Extracting emission line spectra and subtracting from data
    #####################################################################
    if pp.has_emission:
        em_weights = pp.weights[-3:]
        em_matrix = pp.matrix[:,-3:]
        em = em_matrix.dot(em_weights)
    else:
        em = np.zeros_like(pp.bestfit)
    #########################################################################
    # Handle cases where more than one component is used
    if pp.ncomp > 1:
        sol = pp.sol[0]
        error = pp.error[0]
    else:
        sol = pp.sol
        error = pp.error
    ##########################################################################
    if error[1] == 0.0:
        print "Skiped galaxy: unconstrained sigma."
        return
    ##########################################################################
    lick_sim = np.zeros((Nsim, 25))
    vpert = np.random.normal(sol[0], error[0], Nsim)
    sigpert = np.random.normal(sol[1], error[1], Nsim)
    for j in np.arange(Nsim):
        noise_sim = np.random.normal(0, pp.noise, len(pp.bestfit))
        obs_sim = lector.broad2lick(pp.w, pp.bestfit + noise_sim - em,
                                    2.54, vel=vpert[j])
        l, err = lector.lector(pp.w, obs_sim, noise_sim, bands, vel = vpert[j],
                               cols=(0,8,2,3,4,5,6,7), keeplog=0)
        lick_sim[j] = l * bcorr(sigpert[j], l)
    with open(output, "w") as f:
        np.savetxt(f, lick_sim)
    print "Finished MC for {0}.".format(spec)
    return

def table_header():
     return "# Spectra\tHd_A\tHd_F\tCN_1\tCN_2\tCa4227\tG4300\tHg_A\tHg_F" \
           "\tFe4383\tCa4455\tFe4531\tCe4668\tH_beta\tFe5015\tMg_1\tMg_2\t" \
            "Mg_b\tFe5270\t	Fe5335\tFe5406\tFe5709\tFe5782\tNa_D\t" \
             "TiO_1\tTiO_2\n"

def write_table(specs, Nsim):
    results = []
    for i, spec in enumerate(specs):
        outname = "mc_logs/{0}_nsim{1}.txt".format(spec.replace(".fits", ""),
                                                   Nsim)
        output = os.path.join(wdir, outname)
        if not os.path.exists(output):
            continue
        data = np.loadtxt(output)
        median =np.median(data, axis=0)
        mad = 1.4826 * np.median(np.abs(data - median), axis=0)
        line = "{0:26s}".format(spec) + \
               "".join(["{0:10s}".format("{0:.5f}".format(x)) for x in mad])
        results.append(line)
    with open("mc_lick_nsim{0}.txt".format(Nsim), "w") as f:
        f.write("\n".join(results))
    return

if __name__ == "__main__":
    wdir = os.path.join(home, "p5pc")
    outdir = os.path.join(wdir, "mc_logs")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    os.chdir(wdir)
    bcorr = BroadCorr(os.path.join(tables_dir, "lickcorr_m.txt"))
    specs = speclist()
    bands = os.path.join(tables_dir, "BANDS")
    lick_types = np.loadtxt(bands, usecols=(8,))
    lick_indices = np.genfromtxt(bands, usecols=(0,), dtype=None).tolist()
    Nsim = 400
    header = table_header()
    if True:
        pool = mp.Pool()
        for i, spec in enumerate(specs):
            pool.apply_async(run_mc, args=(spec, i))
            # print spec
            # run_mc(spec, i)
            # break
        pool.close()
        pool.join()
    write_table(specs, Nsim)
