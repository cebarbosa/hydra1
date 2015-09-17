# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 17:39:22 2013

@author: cbarbosa

Delete models of MCMC
"""
import os
import shutil

from config import *

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    folders = [x for x in os.listdir(".") if os.path.isdir(x)]
    for folder in folders:
        if folder in ["figs", "logs"]:
            continue
        else:
            shutil.rmtree(folder)

