# -*- coding: utf-8 -*-
"""

Created on 20/05/16

@author: Carlos Eduardo Barbosa

Produces a latex table for the kinematics.

"""
import os
import datetime

import numpy as np
from astropy.coordinates import Angle
from astropy import units

from config import *
import canvas as cv

def make_tex_table():
    workdir = os.path.join(home, "single2")
    os.chdir(workdir)
    table = "ppxf_results.dat"
    specs = np.loadtxt(table, usecols=(0,), dtype=str)
    ids = [x.split("n3311")[1][:-5].replace("_", " ").upper() for x in specs]
    ##########################################################################
    # Get location of the slits
    coords = get_coords(specs)
    # Version with astropy
    ra = [str(x) for x in Angle(coords[:,0], unit=units.hour)]
    dec = [str(x) for x in Angle(coords[:,1], unit=units.degree)]
    # Version without astropy
    # ra, dec = coords.T
    x,y = get_positions(specs).T
    r = np.sqrt(x*x + y*y)
    pa = np.rad2deg(np.arctan2(x, y))
    ##########################################################################
    # Initialize final array
    out = np.column_stack((ids, ra, dec, r, pa))
    # Get kinematics
    data = np.loadtxt(table, usecols=(1,3,5,7,10,11,12))
    errs =  np.loadtxt(table, usecols=(2,4,6,8))
    for i in range(4):
        out = np.column_stack((out, make_str_err(data[:,i], errs[:,i])))
    sn = np.array(["{0:.1f}".format(x) for x in data[:,4]])
    adegree = np.array(["{0:.0f}".format(x) for x in data[:,5]])
    mdegree = np.array(["{0:.0f}".format(x) for x in data[:,6]])
    out = np.column_stack((out, sn, adegree, mdegree))
    out = [" & ".join(x) + "\\\\" for x in out]
    with open("kinematics.tex", "w") as f:
        f.write("\n".join(out))


def get_positions(specs):
    """ Retrieve position of the slits based on spectrum name. """
    xy = []
    for i, spec in enumerate(specs):
        slit = spec.split(".")[0].split("_", 1)[1][5:]
        index = canvas.slits.ids.index(slit)
        xy.append([canvas.slits.x[index], canvas.slits.y[index]])
    return np.array(xy)

def get_coords(specs):
    """ Get RA and DEC of the slits"""
    slits = cv.Slitlets()
    coords = []
    for spec in specs:
        region = spec.replace(".fits", "").split("_", 1)[1][5:]
        index = slits.ids.index(region)
        coords.append([slits.ra[index],  slits.dec[index]])
    return np.array(coords)

def make_str_err(vals, errs):
    strs = []
    for val, err in zip(vals, errs):
        if not np.isfinite(val) or not np.isfinite(err):
            strs.append("--")
        elif err == 0.:
            strs.append(r"[{0}]".format(val))
        elif err >= 1:
            strs.append(r"${0}\pm{1}$".format(int(round(val)), int(round(err))))
        else:
            ndig = int(np.ceil(np.abs(np.log10(err))))
            ndig = ".{0}".format(ndig)
            strs.append(r"${0:{2}f}\pm{1:{2}f}$".format(val, err,ndig))
    return strs



if __name__ == "__main__":
    canvas = cv.CanvasImage("vband")
    make_tex_table()