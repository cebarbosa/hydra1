# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 20:02:09 2014

@author: cbarbosa

Define cube law colorbar and register in the system. 
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
import brewer2mpl
from scipy.interpolate import interp1d

from setup_n3311 import *

def cubelaw():
    """ Definition of the Cube Law Rainbow colormap"""
    grb = np.loadtxt(os.path.join(tables_dir, "cube1.csv"))
    r = np.linspace(0., 1., 256)
    cdict = {"red" : tuple(zip(r, grb[:,0], grb[:,0])), 
             "green" : tuple(zip(r, grb[:,1], grb[:,1])),
             "blue" : tuple(zip(r, grb[:,2], grb[:,2]))}
    cmap = LinearSegmentedColormap("cubelaw", cdict, 256)
    cm.register_cmap(name='cubelaw', cmap=cmap)
    grb = grb[::-1]
    cdict = {"red" : tuple(zip(r, grb[:,0], grb[:,0])), 
             "green" : tuple(zip(r, grb[:,1], grb[:,1])),
             "blue" : tuple(zip(r, grb[:,2], grb[:,2]))}
    cmap = LinearSegmentedColormap("cubelaw_r", cdict, 256)
    cm.register_cmap(name='cubelaw_r', cmap=cmap)
    return

def newgray():
    """ Modified version of Oranges."""
    oranges = cm.get_cmap("gray", 100)
    array = oranges(np.arange(100))
    array = array[40:]
    cmap = LinearSegmentedColormap.from_list("newgray", array)
    cm.register_cmap(name='newgray', cmap=cmap)
    array = array[::-1]
    cmap = LinearSegmentedColormap.from_list("newgray_r", array)
    cm.register_cmap(name='newgray_r', cmap=cmap)
    return

def cmap_map(function,cmap):
    """ Applies function (which should operate on vectors of shape 3:
    [r, g, b], on colormap cmap. This routine will break any discontinuous
    points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red','green','blue'):
        step_dict[key] = map(lambda x: x[0], cdict[key])
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(map( reduced_cmap, step_list))
    new_LUT = np.array(map( function, old_LUT))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i,key in enumerate(('red','green','blue')):
        this_cdict = {}
        for j,step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j,i]
            elif new_LUT[j,i]!=old_LUT[j,i]:
                this_cdict[step] = new_LUT[j,i]
        colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
        colorvector.sort()
        cdict[key] = colorvector

    return LinearSegmentedColormap('colormap',cdict,1024)

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

       cmap: colormap instance, eg. cm.jet.
       N: number of colors.

       Example
       x = resize(arange(100), (5,100))
       djet = cmap_discretize(cm.jet, 5)
       imshow(x, cmap=djet)
    """
    if type(cmap) == str:
        cmap = ScalarMappable.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [(indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])\
                       for i in xrange(N+1)]
    # Return colormap object.
    return LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

cubelaw()
if __name__ == "__main__":
    fig = plt.figure()
    plt.pcolor(np.reshape(np.linspace(0,100,100*100), (100,100)),
               cmap="RdYlGn")
    plt.colorbar()
    plt.pause(0.001)
    plt.show(block=1)
    
    