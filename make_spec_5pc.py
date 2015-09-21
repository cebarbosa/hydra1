#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on 03/03/15 14:17

@author: Carlos Eduardo Barbosa

"""
import os

import numpy as np
import pyfits as pf
import matplotlib.pyplot as plt

from config import *

if __name__ == "__main__":
    os.chdir(data_dir)
    specs = [x for x in os.listdir(".") if x.endswith(".fits")]
    fig = plt.figure()
    pc = 0.05
    for spec in specs:
        name = spec.replace("fin1", "sky1")
        sky = os.path.join(home, "sky/1D/{0}".format(name))
        if not os.path.exists(sky):
            continue
        data = pf.getdata(spec)
        sdata = pf.getdata(sky)
        header = pf.getheader(spec, verify="ignore")
        outplus = os.path.join(home, "p5pc", spec)
        plusdata = data + pc * sdata

        outminus = os.path.join(home, "m5pc", spec)
        minusdata = data - pc * sdata
        ax = plt.subplot(111)
        ax.plot(sdata, "-g")
        ax.plot(data, "-k")
        ax.plot(plusdata, "-b")
        # ax.plot(minusdata, "-r")
        # ax.set_xlim(360,480)
        ax.set_ylim(0,150)
        plt.pause(1)
        plt.show(block=0)
        if False:
            pf.writeto(outplus, plusdata, clobber=True)
            pf.writeto(outminus, minusdata, clobber=True)
        plt.cla()



