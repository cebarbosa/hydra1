# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:01:09 2013

@author: cbarbosa

Make general tools for produce maps
"""
import os
import fileinput

import pywcs
import numpy as np
import pyfits as pf
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
from matplotlib.path import Path

import matplotlib.patches as patches

from config import *

class CanvasImage():
    """ This class defines the properties of background image to be used in 
        maps and also the position of the slits in the image. """
        
    def __init__(self, im):
        """ Set attributes of canvas image. 
        
        ----------------
        Input Parameters
        ----------------
        image : string
            Image to be used. Options are "vband", "residual", "dss" and 
            "xrays". Other images can be registered in the function set_input.
        """
        self.imtype = im
        self.set_input()
        self.D = 50.7 # Mpc
        self.data = pf.getdata(self.image)
        self.header = pf.getheader(self.image)
        self.wcs = pywcs.WCS(self.header)
        self.set_center()
        self.calc_extent()
        self.slits = Slitlets()
        self.slit_arrays()
        self.rescale()
    
    def arcsec2kpc(self, x):
        """ Conversion between angular units. """
        return 2 * self.D * 1000. * np.tan(np.deg2rad(x / 3600. / 2.))
        
    def calc_extent(self):
        """Calculate extension of image according to image. """
        ysize, xsize = self.data.shape
        pixcrd = np.array([[1,1], [ysize, xsize]])
        # For some reason pywcs is not making things right here for vband image
        if self.imtype in ["vband", "residual"]:
            ra, dec = [[159.2268, 159.0858], [-27.5978, -27.4792]]
        if self.imtype in ["xrays"]:
            ra, dec = np.array([[159.6196417, 158.8087792],
                                [-27.9140444, -27.19518333]])
            ra += -0.002
        self.extent_arcsec = 3600 * np.array([ra[0] - self.ra0,
            ra[1] - self.ra0, dec[0] - self.dec0, dec[1] - self.dec0])
        self.extent = self.arcsec2kpc(self.extent_arcsec)

    def calc_vertices(self, x, y, w, h, ang):
        """ Return the vertices of the rectangle in units of pixels. """
        offset = np.column_stack((x, y))
        vertices = np.zeros((len(x), 5, 2))
        for i, (ww, hh, aa) in enumerate(np.column_stack((w, h, ang))):
            theta = (np.pi / 180.0) * (aa + self.posangle)
            R = np.array([[np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]])
            rect = np.array([(-ww, -hh), (-ww, hh), (ww, hh), 
                             (ww, -hh), (-ww, -hh)])/2.
            vertices[i] = np.dot(rect, R) + offset[i]
        return vertices
    
    def draw_arrows(self, c="y", x=-48, y=-48, l=10):
        """ Make N-E arrows"""
        plt.arrow(x, y, 0, l, fc=c, ec=c, 
                  head_width=l/10., head_length=l/5.)
        plt.arrow(x, y, l, 0, fc=c, ec=c,
                  head_width=l/10., head_length=l/5.)
        plt.annotate("N", (x + 1.9, y + l + 2.5), color=c, size=15)
        plt.annotate("E", (x+l + 5.5 ,y-1.5), color=c, size=15)
        return

    def draw_literature(self):
        """ Draw slits from the literature: Ventimiglia et al. 2010 and 
            Richtler et al. 2011. """
        ra = np.array([159.1910417, 159.1780833, self.ra0, self.ra0])
        dec = np.array([-27.52338889, -27.52813889, self.dec0, self.dec0])
        xs = self.arcsec2kpc(3600 * (ra - self.ra0))
        ys = self.arcsec2kpc(3600 * (dec - self.dec0))
        w = self.arcsec2kpc(np.array([3, 3, 3, 3]))
        l = self.arcsec2kpc(np.array([150, 150, 250, 250]))
        theta = -np.array([142, 63, 40, 115])
        vertices = self.calc_vertices(xs, ys, w, l, theta)
        cs = ["b", "b", "g", "g"]
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                 Path.CLOSEPOLY,]
        for i, rect in enumerate(vertices):  
#            if i != 0:
#                continue
            path = Path(rect, codes)
            patch = patches.PathPatch(path, facecolor=cs[i], lw=0.)
            ax.add_patch(patch) 
        return

    def draw_slits(self, ax, slit_type=1, fc="none", ec="k",
                   amp=1., ignore=None):
        """ Draw slits of a given a numerical type. """
        if ignore is None:
            ignore = []
        ids = np.where(self.slits.type == slit_type)[0]
        rects = amp * self.calc_vertices(self.slits.x[ids], self.slits.y[ids],
                                   self.slits.w[ids], self.slits.l[ids], 
                                   self.slits.ang[ids] + self.posangle)
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                 Path.CLOSEPOLY,]
        for i, (rect, id) in enumerate(zip(rects, ids)):
            if self.slits.ids[id] in ignore:
                print "Ignoring", self.slits.ids[id]
                continue
            path = Path(rect, codes)
            patch = patches.PathPatch(path, facecolor=fc, edgecolor=ec,
                                      alpha=1., linewidth=.3)
            ax.add_patch(patch)
        return
    
    def draw_slits_masks(self, inds, c = "r"):
        """ Draw slits by mask."""
        rects = self.calc_vertices(self.slits.x[ids], self.slits.y[ids],
                                   self.slits.w[ids], self.slits.l[ids], 
                                   self.slits.ang[ids] + self.posangle)
        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO,
                 Path.CLOSEPOLY,]
        for i, rect in enumerate(rects): 
            path = Path(rect, codes)
            patch = patches.PathPatch(path, facecolor=c, edgecolor=c)
            ax.add_patch(patch)
            name = self.slits.ids[ids[i]].split("_")[1][1:]
            x = self.slits.x[ids[i]]
            y = self.slits.y[ids[i]]
            if name.endswith("b"):
                continue
            if name.endswith("a"):
                name = name[:-1]
            ax.annotate(name, (x,y), color=c, size=10)

    def get_positions_by_slits(self, ids):
        """ Get position of slits by the identification of slits """
        xy = []
        for i, idd in enumerate(ids):
            index = canvas.slits.ids.index(idd)
            xy.append([self.slits.x[index], self.slits.y[index]])
        return np.array(xy)


    def imshow(self, cmap="cubehelix", **kwargs):
        plt.imshow(self.data, origin="bottom", cmap=cmap, 
                   extent=self.extent, **kwargs)

    def lims(self, radius=40):
        plt.xlim(radius, -radius)
        plt.ylim(-radius, radius)

    def make_contours(self, color="k"):
        """ Overplot countours of image. """
        if self.imtype == "residual":
            self.contours = np.linspace(22, 25.5, 8)
            nsig = 5.
        elif self.imtype == "vband":
            self.contours = np.linspace(19,23.5,10)
            nsig = 3.
        datasmooth = ndimage.gaussian_filter(self.data, nsig, order=0.)
        plt.contour(datasmooth, self.contours, extent=self.extent, 
                    colors=color, linewidths=1.0)
        return

    def rescale(self):
        """ Rescale images to work in surface brightness units. """
        if self.imtype == "residual":
            self.data = np.clip(self.data, 1., self.data.max())
            self.data = -2.5 * np.log10(self.data/480./self.ps**2) +  27.2
            yc, xc, r = 775, 1251, 90
            for x in np.arange(xc-r, xc+r):
                for y in np.arange(yc-r, yc+r):
                    if (x - xc)**2 + (y - yc)**2 < r**2:
                        self.data[x,y] = np.nan
        elif self.imtype == "vband":
            self.data = np.clip(self.data - 4900., 1., self.data)
            self.data = -2.5 * np.log10(self.data/480./self.ps/self.ps) + 27.2
        
    def set_input(self):
        """ Set image for using as the canvas for the maps. 
        
        Here you can add new images for using as background of images. 
        
        """
        if self.imtype == "vband":
            self.image = os.path.join(images_dir,  "Vband_large.fits")
            self.ps = 0.252 
            self.posangle = 0.
        elif self.imtype == "residual":
            self.image = os.path.join(images_dir,  "Vband_residual.fits")
            self.ps = 0.252 
            self.posangle = 0.
        elif self.imtype == "xrays":
            self.image = os.path.join(images_dir,  "xray.fits")
            self.ps = 4.1
            self.posangle = 0.
        elif self.imtype == "galexuv":
            self.image = os.path.join(images_dir,  "AIS_329_sg12-fd-int.fits")
            self.ps = 1
            self.posangle = 0.
        return 
            
    def set_center(self):
        """ Set the central coordinate properties. """
        self.ra0 = 159.1783
        self.dec0 = -27.52833
        center = self.wcs.wcs_sky2pix([[self.ra0]], [[self.dec0]], 1)
        self.xc, self.yc = center[0][0], center[1][0]
        return
        
    def slit_arrays(self):
        """ Calculate arrays of slit-related properties and convert to kpc. """
        self.slits.xpix, self.slits.ypix = self.wcs.wcs_sky2pix(self.slits.ra, 
                                                    self.slits.dec, 1)
        self.slits.xarcsec = 3600. * (self.slits.ra - self.ra0)
        self.slits.yarcsec = 3600. * (self.slits.dec - self.dec0)
        self.slits.x = self.arcsec2kpc(self.slits.xarcsec)
        self.slits.y = self.arcsec2kpc(self.slits.yarcsec)
        
        self.slits.radius, self.slits.pa = cart2polar(self.slits.x, 
                                                      self.slits.y)
        self.slits.w = self.arcsec2kpc(self.slits.w)
        self.slits.l = self.arcsec2kpc(self.slits.l)
        self.slits.vertices = self.calc_vertices(self.slits.x, 
                              self.slits.y, self.arcsec2kpc(self.slits.w), 
                              self.arcsec2kpc(self.slits.l), 
                              ang=self.slits.pa)
        return
        

class Slitlets():
    """ Class to define the object slitlets of the Hydra I observations. """
    def __init__(self):
        self.load_files()
        self.set_ids()
        self.set_arrays()
    
    def load_files(self):
        fldr = os.path.join(tables_dir,"reftables2")
        files = [os.path.join(fldr, x) for x in os.listdir(fldr)]
        table = []
        for line in fileinput.input(files):
            table.append(line)
        self.table = table
        return
    
    def set_ids(self):
        nums = np.genfromtxt(self.table, usecols=(0,), dtype=None).tolist()
        mask = np.genfromtxt(self.table, usecols=(5,), dtype=None).tolist()
        self.ids = ["{0}_s{1}".format(m[5:],n) for m,n in zip(mask,nums)]
        return
    
    def set_arrays(self):
        self.ra, self.dec, self.l = np.loadtxt(self.table, 
                                                    usecols=(1,2,6)).T        
        self.type = np.loadtxt(self.table, usecols=(7,))
        self.ang = -40. * np.ones_like(self.l) 
        self.w = np.ones_like(self.l) 
        
def cart2polar(x, y):
    """ Simple coordinate transformation from Cartesian to Polar."""
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)
    return r, np.rad2deg(theta)

def circle_xy(r, phi=None):
    """ Make array of (x,y) coordinates of a circle of radius r."""
    if phi == None:
        phi = np.arange(0, 6.28, 6.28/80.)
    return np.column_stack((r * np.cos(phi), r * np.sin(phi)))

if __name__ == "__main__":
    pass