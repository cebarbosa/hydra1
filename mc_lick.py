# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:22:29 2014

@author: cbarbosa

Make simulations to derive errors on Lick indices including both noise and the 
error on the determined velocity.
"""
import numpy as np
import multiprocessing as mp
from scipy.interpolate import interp1d

from setup_n3311 import *
import lector
from lick_hydra1_vdispcorr import BroadCorr
from run_ppxf import pPXF

def run_mc(spec, i):
    """ Run MC routine in single spectrum. """
    print "{0} ({1}/{2})".format(spec, i+1, len(specs))
    # Check if MC was done previously
    # if spec in specs_done:
    #     return
    # Avoid problematic cases
    # if sigerr == 0:
    #     print "Skiped MC for {0}.".format(spec)
    #     return
    pp = pPXF(spec, velscale)
    #####################################################################
    # Extracting emission line spectra and subtracting from data
    #####################################################################
    if pp.has_emission:
        em_weights = pp.weights[-3:]
        em_matrix = pp.matrix[:,-3:]
        em = em_matrix.dot(em_weights)
    else:
        em = np.zeros_like(pp.bestfit)
    ######################################################################
    # Handle cases where more than one component is used
    if pp.ncomp > 1:
        sol = pp.sol[0]
        error = pp.error[0]
    else:
        sol = pp.sol
        error = pp.error
    lick_sim = np.zeros((Nsim, 25))
    vpert = np.random.normal(sol[0], error[0], Nsim)
    sigpert = np.random.normal(sol[1], error[1], Nsim)
    # try:
    for j in np.arange(Nsim):
        noise_sim = np.random.normal(0, pp.noise, len(pp.bestfit))
        obs_sim = lector.broad2lick(pp.w, pp.bestfit + noise_sim - em, 2.54, vel=vpert[j])
        l, err = lector.lector(pp.w, obs_sim, noise_sim, bands, vel = vpert[j],
                               cols=(0,8,2,3,4,5,6,7), keeplog=0)
        lick_sim[j] = l * bcorr(sigpert[j], l)
    sim_err = list(np.std(lick_sim, axis=0))
    sim_err = spec + "\t" + "\t".join([str(x) for x in sim_err]) + "\n"
    with open(output, "a") as f:
        f.write(sim_err)
    # except:
    #     pass
    print "Finished MC for {0}.".format(spec)
    return

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    bcorr = BroadCorr(os.path.join(tables_dir, "lickcorr_m.txt"))
    specs = np.loadtxt("ppxf_results.dat", usecols=(0,), dtype=str).tolist()
    vels, velerrs = np.loadtxt("ppxf_results.dat", usecols=(1,2)).T
    sigmas, sigmaerrs = np.loadtxt("ppxf_results.dat", usecols=(3,4)).T
    bands = os.path.join(tables_dir, "BANDS")
    lick_types = np.loadtxt(bands, usecols=(8,))
    lick_indices = np.genfromtxt(bands, usecols=(0,), dtype=None).tolist()
    Nsim = 400
    output = "lick_mc_errs_{0}.txt".format(Nsim)
    specs_done = []
    if os.path.exists(output):
        specs_done = np.loadtxt(output, usecols=(0,), dtype=str).tolist()
    else:
        with open(output, "a") as f:
            f.write("# Monte Carlo simulation of the errors\n")
            f.write("# Spectra\tHd_A\tHd_F\tCN_1\tCN_2\tCa4227\tG4300\tHg_A\tHg_F"
                    "\tFe4383\tCa4455\tFe4531\tCe4668\tH_beta\tFe5015\tMg_1\tMg_2\t"
                    "Mg_b\tFe5270\t	Fe5335\tFe5406\tFe5709\tFe5782\tNa_D\t"
                    "TiO_1\tTiO_2\n")
    pool = mp.Pool()
    for i, spec in enumerate(specs):
        if spec in specs_done:
            continue
        pool.apply_async(run_mc, args=(spec, i))
        # print spec
        # run_mc(spec, i)
    pool.close()
    pool.join()
