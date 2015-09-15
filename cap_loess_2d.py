###############################################################################
#
# Copyright (C) 2010-2014, Michele Cappellari
# E-mail: cappellari_at_astro.ox.ac.uk
#
# Updated versions of the software are available from my web page
# http://purl.org/cappellari/software
#
# If you have found this software useful for your research,
# I would appreciate an acknowledgment to the use of the
# "CAP_LOESS_2D routine of Cappellari et al. (2013b), which implements 
# the multivariate LOESS algorithm of Cleveland & Devlin (1988)"
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
###############################################################################
#+
# NAME:
#       CAP_LOESS_2D
# PURPOSE:
#       Local regression LOESS smoothing http://en.wikipedia.org/wiki/Local_regression
#       Univariate: Cleveland (1979) http://www.jstor.org/stable/2286407
#       Multivariate: Cleveland & Devlin (1988) http://www.jstor.org/stable/2289282
#
# CALLING EXAMPLE:
#       zout = loess_2d(x, y, z, frac=0.5, degree=1, rescale=False)
#           
# INPUT PARAMETERS:
#   X, Y, Z: vectors of equal number of elements containing the
#       coordinates (X,Y) and the corresponding function values Z
#       to be smoothed.
#   ZOUT: vector of smoothed Z values (same size as Z)
#
# KEYWORDS:
#   DEGREE: degree of the local approximation (typically 1 or 2)
#   FRAC: Fraction of points to consider in the local approximation.
#       Typical values are between ~0.2-0.8. Note that the values are
#       weighted by their distance from the point under consideration.
#       This implies that the effective fraction of points contributing 
#       to a given value is much smaller that FRAC.
#   NPOINTS: Number of points to consider in the local approximation.
#       This is an alternative to giving FRAC=NPOINTS/n_elements(x).
#   RESCALE: Rescale the (X,Y) coordinates to transform the inertia 
#       ellipse into a circle. Using this keyword is useful when
#       the X and Y coordinates do not represent the same physical 
#       quantity.
#   SIGZ: 1-sigma errors for the Z values. If this keyword is used
#       the biweight fit is done assuming those errors. If this keyword 
#       is *not* used, the biweight fit determines the errors in Z
#       from the scatter of the neigbouring points.  
#   WOUT: Output weights used in the fit. This can be used to 
#       identify outliers: wout=0 for outliers deviations >4sigma.
#  
# MODIFICATION HISTORY:
#   V1.0: Michele Cappellari Oxford, 15 December 2010
#   V1.1: Rescale after rotating to axis of maximum variance.
#       MC, Vicenza, 30 December 2010
#   V1.11: Fix: use ABS() for proper computation of "r". 
#       MC, Oxford, 07 March 2011
#   V1.12: Return values unchanged if FRAC=0. MC, Oxford, 25 July 2011
#   V1.13: Check when outliers don't change to stop iteration.
#       MC, Oxford, 2 December 2011
#   V1.14: Updated documentation. MC, Oxford, 16 May 2013
#   V1.32: Test whether input (X,Y,Z) have the same size. 
#       Included NPOINTS keyword. MC, Oxford, 12 October 2013
#   V1.33: Use CAP_POLYFIT_2D. Removed /QUARTIC keyword and replaced
#       by DEGREE keyword like CAP_LOESS_1D. MC, Oxford, 31 October 2013
#   V1.34: Include SIGZ and WOUT keywords. Updated documentation. 
#       MC, Paranal, 7 November 2013
#   V2.0: Translated from IDL into Python. MC, Oxford, 26 February 2014
#-
#------------------------------------------------------------------------

import numpy as np
from scipy import linalg

#------------------------------------------------------------------------

def polyfit_2d(x, y, z, degree, sigz=None, weights=None):

    if weights is None:
        if sigz is None:
            sw = 1.
        else:
            sw = 1./sigz
    else:
        sw = np.sqrt(weights)
    
    npol = (degree+1)*(degree+2)/2
    a = np.zeros((x.size, npol))
    c = a.copy()
    k = 0
    for i in range(degree+1):
        for j in range(degree-i+1):
            c[:,k] = x**j * y**i
            a[:,k] = c[:,k]*sw
            k += 1

    coeff = linalg.lstsq(a, z*sw)[0]
    
    return c.dot(coeff)

#----------------------------------------------------------------------------------

def biweight_sigma(y, zero=False):
    """
    Biweight estimate of the scale (standard deviation).
    Implements the approach described in 
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983, Chapter 12B, pg. 417
    
    """
    y = np.ravel(y)
    if zero:
        d = y
    else:
        d = y - np.median(y)
        
    mad = np.median(np.abs(d))
    u2 = (d / (9.0*mad))**2 # c = 9
    good = u2 < 1.0
    u1 = 1.0 - u2[good]
    num = y.size * ((d[good]*u1**2)**2).sum()
    den = (u1*(1.0 - 5.0*u2[good])).sum()
    sigma = np.sqrt(num/(den*(den - 1.0))) # see note in above reference

    return sigma

#----------------------------------------------------------------------------

def biweight_mean(y, itmax=10):
    """
    Biweight estimate of the location (mean).
    Implements the approach described in 
    "Understanding Robust and Exploratory Data Analysis"
    Hoaglin, Mosteller, Tukey ed., 1983
    
    """
    y = np.ravel(y)        
    c = 6.
    fracmin = 0.03*np.sqrt(0.5/(len(y)-1.))    
    y0 = np.median(y)
    mad = np.median(abs(y - y0))
    
    frac = 1e30
    it = 0
    while ((frac > fracmin) and (it < itmax)):
        it += 1
        u2 = ((y - y0)/(c*mad))**2
        u2 = u2.clip(0,1)
        w = (1. - u2)**2
        w /= np.sum(w) 
        mad_old = mad
        y0 += np.sum(w*(y - y0))
        mad = np.median(abs(y - y0))
        frac = abs(mad_old - mad)/mad

    return y0

#------------------------------------------------------------------------

def rotate_points(x, y, ang):
    """
    Rotates points conter-clockwise by an angle ANG in degrees.
    Michele cappellari, Paranal, 10 November 2013
    
    """
    theta = np.radians(ang)
    xNew = x*np.cos(theta) - y*np.sin(theta)
    yNew = x*np.sin(theta) + y*np.cos(theta)
    
    return xNew, yNew
    
#------------------------------------------------------------------------

def loess_2d(x1, y1, z, frac=0.5, degree=1, rescale=False, 
             npoints=None, sigz=None):
    """
    zout = loess_2d(x, y, z, frac=0.5, degree=1)
    gives a LOESS smoothed estimate of the quantity z at the sets 
    of coordinates (x,y).
    
    """

    if frac == 0:
        return z
    
    n = x1.size
    if (n != y1.size) | (n != z.size):
        raise ValueError('Input vectors (X, Y, Z) must have the same size')
    
    if npoints is None:
        npoints = np.ceil(n*frac)
    
    if rescale is True:
    
        # Robust calculation of the axis of maximum variance
        #
        nsteps = 180
        angles = np.arange(nsteps)
        sig = np.zeros(nsteps)
        for j in range(nsteps):
            x2, y2 = rotate_points(x1, y1, angles[j])
            sig[j] = biweight_sigma(x2)
        k = np.argmax(sig) # Find index of max value
        x2, y2 = rotate_points(x1, y1, angles[k])
        x = (x2 - biweight_mean(x2)) / biweight_sigma(x2)
        y = (y2 - biweight_mean(y2)) / biweight_sigma(y2)
        
    else:
    
        x = x1
        y = y1
       
    zout = np.zeros(n)
    wout = zout.copy()
    
    for j in range(n):
    
        dist = np.sqrt((x - x[j])**2 + (y - y[j])**2)
        w = np.argsort(dist)[:npoints]
        distWeights = (1. - (dist[w]/dist[w[-1]])**3)**3 # tricube function distance weights
        zfit = polyfit_2d(x[w], y[w], z[w], degree, weights=distWeights)
    
        # Robust fit from Sec.2 of Cleveland (1979)
        # Use errors if those are known.
        #
        bad = []
        for p in range(10): # do at most 10 iterations
            if sigz is None: # Errors are unknown
                aerr = np.abs(zfit - z[w]) # Note ABS()
                mad = np.median(aerr) # Characteristic scale
                uu = (aerr/(6.*mad))**2 # For a Gaussian: sigma=1.4826*MAD 
                uu = uu.clip(0,1)
            else: # Errors are assumed known 
                uu = ((zfit - z[w])/(4.*sigz[w]))**2
                uu = uu.clip(0,1)
            biWeights = (1. - uu)**2
            totWeights = distWeights*biWeights
            zfit = polyfit_2d(x[w], y[w], z[w], degree, weights=totWeights)
            badOld = bad
            bad = np.where(biWeights < 0.34)[0] # 99% confidence outliers
            if len(badOld) == len(bad):
                if np.all(np.equal(badOld, bad)): 
                    break
        
        zout[j] = zfit[0]
        wout[j] = biWeights[0]
    
    return zout

#------------------------------------------------------------------------
