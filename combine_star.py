# -*- coding: utf-8 -*-
"""

Created on 22/02/16

@author: Carlos Eduardo Barbosa

Combine spectra of standard star HD 102070

"""
import os

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt

from config import *
from run_ppxf import wavelength_array

if __name__ == "__main__":
    os.chdir(os.path.join(home, "data/star"))
    spec2d = pf.getdata("HD102070_2d.fits")
    w = wavelength_array("HD102070_2d.fits")
    star = np.mean(spec2d[1188:1231,:], axis=0)
    h = pf.getheader("HD102070_2d.fits")
    h["NAXIS"] = 1
    del h["NAXIS2"]
    pf.writeto("HD102070_noflux.fits", star, h, clobber=True)


