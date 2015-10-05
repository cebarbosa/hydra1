# -*- coding: utf-8 -*-
"""

Created on 05/10/15

@author: Carlos Eduardo Barbosa

Test convergence of Lick errors as a function of the number of simulation.

"""
import os

import numpy as np
from pandas import expanding_std
import matplotlib.pyplot as plt

from config import *

if __name__ == "__main__":
    os.chdir(os.path.join(home, "single2/mc_logs"))
    cols = np.array([12, 13,16,17,18,19,20])
    logs = os.listdir(".")
    fig = plt.figure(1, figsize=(5,15))
    for log in logs:
        data = np.loadtxt(log).T[cols]
        for i,d in enumerate(data):
            ax = plt.subplot(7,1,i+1)
            ax.plot(expanding_std(d, min_periods=1) / d.std(), "-k")
        plt.pause(1)
        plt.show(block=False)

