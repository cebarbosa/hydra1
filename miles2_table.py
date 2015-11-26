# -*- coding: utf-8 -*-
"""

Created on 26/11/15

@author: Carlos Eduardo Barbosa

Convert table with SSP parameters of MILES II from the original to the
appropriate format for the MCMC run.

"""
import os

import numpy as np

from config import *

if __name__ == "__main__":
    os.chdir(os.path.join(home, "tables"))
    miles2 = "MILES_BaSTI_un_1.30.LICK.txt"
    lick = "BANDS"
    data = np.loadtxt(miles2, dtype=str)
    header = data[0]
    names = data[:,0]
    cols = np.array([2,3,4,5,6,7,8,9,14,15,16,17,18,24,25,26,27,28,29,30,31,
                     32,33,34,35])
    data = data[:,cols]
    # lick = np.loadtxt("BANDS", dtype=str, usecols=(0,))
    # for a in zip(header[cols], lick):
    #     print a
    table = []
    for name, d in zip(names, data):
        Z = name[8:13].replace("m", "-").replace("p", "+")
        age = name[14:21]
        alpha = name[25:29]
        scale = name[30:]
        if scale not in ["Ep0.00", "Ep0.40"]:
            continue
        if float(age) < 1.:
            continue
        table.append(np.hstack((age, Z, alpha, d)))
    table = np.array(table)
    header = np.hstack(("# Age(Gyr)", "[Z/H]", "[alpha/Fe]", header[cols]))
    header = ["{0:12}".format(x) for x in header]
    with open("MILESII.txt", "w") as f:
        f.write("".join(header))
        np.savetxt(f, table, fmt="%12s")


