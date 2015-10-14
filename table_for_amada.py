# -*- coding: utf-8 -*-
"""

Created on 13/10/15

@author: Carlos Eduardo Barbosa

Make table to test AMADA
"""
import os

import numpy as np

from config import *

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2"))
    cols = (1,2,5,7,69,72,75,84)
    a = np.loadtxt("results.tab", usecols=cols)
    fields = ["X", "Y", "V", "SIGMA", "AGE", "METAL", "ALPHA", "IRON"]
    fields = ["{0:15s}".format(x) for x in fields]
    with open("results.txt", "w") as f:
        f.write("".join(fields) + "\n")
        np.savetxt(f, a)
