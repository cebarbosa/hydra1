# -*- coding: utf-8 -*-
"""

Created on 10/11/15

@author: Carlos Eduardo Barbosa

Prepare files for alternative pPXF run.

"""

import os

import pyfits as pf
import matplotlib.pyplot as plt

from config import *
from run_ppxf import speclist

if __name__ == "__main__":
    spec_dir = os.path.join(home, "single1")
    sky_dir = os.path.join(home, "sky/1D")
    for spectrum in speclist():
        print spectrum
        spec = os.path.join(spec_dir, spectrum)
        sky = os.path.join(sky_dir, spectrum.replace("fin", "sky"))
        outdir = os.path.join(home, "single3")
        output = os.path.join(outdir, spectrum)
        if not (os.path.exists(spec) and os.path.exists(sky)):
            continue
        sdata = pf.getdata(spec)
        skydata = pf.getdata(sky)
        header = pf.getheader(spec, verify="ignore")
        sdata += skydata
        ax = plt.subplot(111)
        ax.plot(sdata, "-k")
        ax.plot(sdata - skydata, "-b")
        ax.plot(skydata, "-y")
        ax.set_xlim(360,480)
        ax.set_ylim(0,150)
        plt.pause(1)
        plt.show(block=1)
        # pf.writeto(output, sdata, header=header, clobber=True,
        #            output_verify='ignore')


