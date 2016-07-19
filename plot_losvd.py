# -*- coding: utf-8 -*-
"""

Created on 21/06/16

@author: Carlos Eduardo Barbosa

Produces 1D plots for LOSVD results
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from config import *
from maps_losvd import get_richtler, get_ventimiglia

if __name__ == "__main__":
    wdir = os.path.join(home, "single2")
    intable = "results.tab" # Input table produced with maps_losvd.py
    print get_richtler()
