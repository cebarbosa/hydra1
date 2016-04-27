# -*- coding: utf-8 -*-
"""

Created on 25/04/16

@author: Carlos Eduardo Barbosa

Split reftables to include sub specs

"""

import os
import warnings

import numpy as np
import pyfits as pf
import pywcs

from config import *

def split_hcg007(data):
    id, rac, decc, xpos, ypos, field, leng, flag, ref = data
    rac = float(rac)
    decc = float(decc)
    xpos = float(xpos)
    ypos = float(ypos)
    mask = field[5:]
    spec = "{0}_s{1}".format(mask, id)
    os.chdir(img_dir)
    imgs = [x for x in os.listdir(".") if x.startswith(field)
            and "_slit_" in x]
    h = pf.getheader(imgs[0])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        wcs = pywcs.WCS(h)
    ##########################################################################
    # Center of the sub-slit in relation to the central
    borders = [np.array([1., 17 - 5.7, 17 + 5.5, 35.]),
               np.array([1., 21 - 6.2, 21 + 6.2, 40.]),
               np.array([1., 36 - 9., 36 - 3., 36 + 3., 36 + 9., 63.])]
    deltay = []
    sizes = []
    for a in borders:
        center = 0.5 * (a[-1] - a[0])
        deltay.append(a[:-1] + 0.5 * np.diff(a) - center)
        sizes.append(0.25 * np.diff(a))
    slits = ["cen1_s14", "cen2_s45", "inn2_s39"]
    deltay = dict(zip(slits, deltay))
    snum = [[2,1,3], [3, 1, 2], [5, 3, 1, 2, 4]]
    snum = dict(zip(slits, snum))
    sizes = dict(zip(slits, sizes))
    ##########################################################################
    dy = deltay[spec]
    length = sizes[spec]
    pixxy = np.column_stack((xpos * np.ones_like(dy), ypos + dy))
    ra, dec = wcs.wcs_pix2sky(pixxy, 1).T
    ra0, dec0 = wcs.wcs_pix2sky([[xpos, ypos]], 1).T
    raslit = rac + (ra - ra0)
    decslit = decc + (dec - dec0)
    lines = []
    for i, (ra, dec) in enumerate(zip(raslit, decslit)):
        newid = "{0}.000{1}".format(id, snum[spec][i])
        newline = "{0} {1:.3f} {2:.4f} {3:.4f} {4:.4f} {5} {6:.1f} {7} " \
                  "{8}".format(newid, ra, dec, xpos, ypos + dy[i], field,
                               sizes[spec][i], flag, ref)
        lines.append(newline.split())
    return np.array(lines)

if __name__ == "__main__":
    reftables_dir = os.path.join(tables_dir, "reftables1")
    output_dir = os.path.join(tables_dir, "reftables2")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    img_dir = "//home/kadu/Dropbox/hydra1/data/slitimages"
    os.chdir(img_dir)
    masks = ["cen1", "cen2", "inn1", "inn2", "out1", "out2"]
    for mask in masks:
        table = os.path.join(reftables_dir, "n3311{0}.txt".format(mask))
        outtable = os.path.join(output_dir, "n3311{0}.txt".format(mask))
        with open(table) as f:
            head = f.readline()
            data = np.loadtxt(f, dtype=str)
        ######################################################################
        # Append new lines
        idx = np.where(data[:,7]=="3")[0] # Find indices for HCG 007
        if len(idx) > 0:
            newlines = split_hcg007(data[idx[0]])
            # Comment old line
            data[idx[0],0] = "#" + data[idx[0],0]
            data = np.vstack((data, newlines))
        with open(outtable, "w") as f:
            f.write(head)
            np.savetxt(f, data, fmt="%s")