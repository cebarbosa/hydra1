# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 17:49:32 2014

@author: cbarbosa

Make LateX table for the population paper.
"""
import os

import numpy as np

from config import *
from mcmc_model import get_model_lims

def ra2str(ra):
    ra /= 15.
    hours = ra.astype(int)
    minutes = ((ra - hours) * 60.)
    minutes = minutes.astype(int)
    seconds = (3600*ra - hours * 3600 - minutes * 60)
    seconds = np.around(seconds, 2)
    rastr = []
    for h,m,s in zip(hours, minutes, seconds):
        if s < 10:
            sfill = "0"
        else:
            sfill = ""
        rastr.append(r"{0}:{1}:{3}{2}".format(h, m, s, sfill))
    return rastr


def dec2str(dec):
    sign=2*np.sign(dec) - 1
    signstr = []
    for sg in sign:
        signstr.append("+" if sg == 1 else "-")
    dec = np.abs(dec)
    hours = dec.astype(int)
    minutes = (dec - hours) * 60
    minutes = minutes.astype(int)
    seconds = (3600*dec - hours * 3600 - minutes * 60)
    seconds = np.around(seconds, 2)
    decstr = []
    for sig, h,m,s in zip(signstr, hours, minutes, seconds):
        if s < 10:
            sfill = "0"
        else:
            sfill = ""
        decstr.append(r"{3}{0}:{1}:{4}{2}".format(h, m, s, sig, sfill))
    return decstr
 
def val2str(s, serr):
    """ Convert value and error to string in LateX."""
    vals = []
    for v, verr in zip(s, serr):
        if np.isnan(v) or np.isnan(verr):
            vals.append(r"--")
        elif np.log10(verr) < 1: 
            sigfig = -int(np.floor(np.log10(verr)))
            newerr = np.around(verr, sigfig)
            if newerr == np.power(10., -sigfig):
                sigfig += 1
                newerr = np.around(verr, sigfig)
            v = np.around(v, sigfig)
            vals.append(r"${0}\pm{1}$".format(v, newerr))
        else:
            vals.append(r"${0}\pm{1}$".format(int(np.around(v)), 
                        int(np.around(verr)))) 
    return vals

def pm_string(pp, ppm, ppp):
    table = []
    for (p, pm, pp) in zip(pp, ppm, ppp):
        line = []
        for i in range(len(p)):
            s = r"${0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}$".format(p[i], pm[i], pp[i])
            line.append(s)
        table.append(" & ".join(line))
    return table
    
if __name__ == "__main__":
    spectype = "single2"
    table = os.path.join(home, spectype, "results.tab")
    spec = np.genfromtxt(table, dtype=None, usecols=(0,))
    ids = [x.split("n3311")[-1].replace(".fits", "").replace("_", " ") for x \
           in spec]
    sns =  np.loadtxt(table, usecols=(14,))
    rs, pas = np.loadtxt(table, usecols=(3,4)).T
    # idx = np.where(sns > sn_cut)[0]
    cols = np.array([39,41,47,49,51,53,55])
    model_table = os.path.join(tables_dir, "models_thomas_2010.dat")
    lims, ranges = get_model_lims(model_table)
    idx = np.array([12,13,16,17,18,19])
    cols2 = np.array([69,72,75])
    lims = lims[idx]
    data = np.loadtxt(table, usecols=cols)
    errs = np.loadtxt(table, usecols=cols+1)
    pop = np.loadtxt(table, usecols=cols2)
    popm = pop - np.loadtxt(table, usecols=cols2 + 1)
    popp = np.loadtxt(table, usecols=cols2 + 2) - pop
    pstring = pm_string(pop, popm, popp)
    for i in range(len(data)):
        for j in range(len(lims)):
            if data[i,j] < lims[j,0] or data[i,j] > lims[j,1]:
                data[i,j] = np.nan
    results = []
    for iid, d, err, r, pa, sn, p in zip(ids, data, errs, rs, pas, sns, pstring):
        if sn < 5:
            continue
        s = "{0} & {1:.1f} & {2:.1f} & {3:.1f} & ".format(iid, r, pa, sn) + \
            " & ".join(val2str(d, err)) + " & " + p
        results.append(s) 
    output = os.path.join(home, spectype, "lick.tex")
    print output
    results.sort()
    with open(output, "w") as f:
        f.write("\\\\\n".join(results) + "\\\\")
