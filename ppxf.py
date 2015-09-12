##############################################################################
#
# Copyright (C) 2001-2013, Michele Cappellari
# E-mail: cappellari_at_astro.ox.ac.uk
#
# Updated versions of the software are available from my web page
# http://purl.org/cappellari/software
#
# If you have found this software useful for your research,
# I would appreciate an acknowledgment to the use of the
# "Penalized Pixel-Fitting method by Cappellari & Emsellem (2004)".
#
# This software is provided as is without any warranty whatsoever.
# Permission to use, for non-commercial purposes is granted.
# Permission to modify for personal or internal use is granted,
# provided this copyright and disclaimer are included unchanged
# at the beginning of the file. All other rights are reserved.
#
##############################################################################
#+
# NAME:
#   ppxf()
#
# PURPOSE:
#   Extract galaxy stellar kinematics (V, sigma, h3, h4, h5, h6)
#   or the stellar population by fitting a template to an observed
#   spectrum in pixel space, using the Penalized Pixel-Fitting (pPXF)
#   method described in: Cappellari M., & Emsellem E., 2004, PASP, 116, 138.
#
#   The following key optional features are also available:
#   1) An optimal template, positive linear combination of different
#      input templates, can be fitted together with the kinematics.
#   2) One can enforce smoothness on the template weights during the fit.
#      This is useful to attach a physical meaning to the weights
#      e.g. in terms of the star formation history of a galaxy.
#   3) One can fit multple kinematic components for both the stars and the gas
#      emission lines. Both the stellar and gas LOSVD can be penalized
#      and can be described by a general Gauss-Hermite series.
#   4) Additive and/or multiplicative polynomials can be included to adjust
#      the continuum shape of the template to the observed spectrum.
#   5) Iterative sigma clipping can be used to clean the spectrum.
#   6) It is possible to fit a mirror-symmetric LOSVD to two spectra at
#      the same time. This is useful for spectra taken at point-symmetric
#      spatial positions with respect to the center of an equilibrium
#      stellar system.
#   7) One can include sky spectra in the fit, to deal with cases
#      where the sky dominates the observed spectrum and an accurate
#      sky subtraction is critical.
#   8) One can derive an estimate of the reddening in the spectrum.
#
# CALLING SEQUENCE:
#
#   import cap_ppxf
#
#   pp = cap_ppxf.ppxf(templates, galaxy, noise, velScale, start, 
#                      bias=None, clean=False, degree=4, goodpixels=None, 
#                      mdegree=0, moments=2, oversample=False, plot=False, 
#                      quiet=False, sky=None, vsyst=0, regul=0, lam=None, 
#                      reddening=None, component=0, reg_dim=None)
#
#  NOTE:
#  The documentation below is taken from the IDL version, which is functionally
#  identical to the Phython version. The various parameters below have the same 
#  meaning here as in the IDL version, but the documentation has not yet been 
#  updated for Python.
#
# INPUT PARAMETERS:
#   TEMPLATES: vector containing the spectrum of a single template star or more
#       commonly an array of dimensions TEMPLATES[nPixels,nTemplates] containing
#       different templates to be optimized during the fit of the kinematics.
#       nPixels has to be >= the number of galaxy pixels.
#     - To apply linear regularization to the WEIGHTS via the keyword REGUL,
#       TEMPLATES should be an array of two TEMPLATES[nPixels,nAge], three
#       TEMPLATES[nPixels,nAge,nMetal] or four TEMPLATES[nPixels,nAge,nMetal,nAlpha]
#       dimensions, depending on the number of population variables one wants to study.
#       This can be useful to try to attach a physical meaning to the output WEIGHTS, in
#       term of the galaxy star formation history and chmemical composition distribution.
#       In that case the templates may represent single stellar population SSP models
#       and should be arranged in sequence of increasing age, metallicity or alpha along 
#       the second, third or fourth dimension of the array respectively.
#     - TEMPLATES and GALAXY do not need to span the same wavelength range. However
#       an error will be returned by PPXF, if the velocity shift in pixels,
#       required to match the galaxy with the templates, becomes larger than
#       nPixels. In that case one has to truncate either the galaxy or the
#       templates to make the two rest-frame spectral ranges more similar.
#   GALAXY: vector containing the spectrum of the galaxy to be measured. The
#       star and the galaxy spectra have to be logarithmically rebinned but the
#       continuum does *not* have to be subtracted. The rebinning may be
#       performed with the LOG_REBIN routine that is distributed with PPXF.
#     - For high redshift galaxies, one should bring the spectra close to the
#       restframe wavelength, before doing the PPXF fit, to prevent too large
#       velocity shifts of the templates. This can be done by dividing the
#       observed wavelenghts by (1+z), where z is a rough estimate of the
#       galaxy redshift, before the logarithmic rebinning.
#     - GALAXY can also be an array of dimensions GALAXY[nGalPixels,2] containing
#       two spectra to be fitted, at the same time, with a reflection-symmetric
#       LOSVD. This is useful for spectra taken at point-symmetric spatial
#       positions with respect to the center of an equilibrium stellar system.
#       For a discussion of the usefulness of this two-sided fitting
#       see e.g. Section 3.6 of Rix & White (1992, MNRAS, 254, 389).
#     - IMPORTANT: 1) For the two-sided fitting the VSYST keyword has to be used.
#       2) Make sure the spectra are rescaled to be not too many order of
#       magnitude different from unity, to avoid over or underflow problems
#       in the calculation. E.g. units of erg/(s cm^2 A) may cause problems!
#   NOISE: vector containing the 1*sigma error (per pixel) in the galaxy spectrum.
#       If GALAXY is a Nx2 array, NOISE has to be an array with the same dimensions.
#     - IMPORTANT: the penalty term of the pPXF method is based on the *relative*
#       change of the fit residuals. For this reason the penalty will work as
#       expected even if no reliable estimate of the NOISE is available
#       (see Cappellari & Emsellem [2004] for details).
#       If no reliable noise is available this keyword can just be set to:
#           NOISE = galaxy*0+1 # Same weight for all pixels
#   VELSCALE: velocity scale of the spectra in km/s per pixel. It has to be the
#       same for both the galaxy and the template spectra.
#   START: two elements vector [velStart,sigmaStart] with the initial estimate
#       for the velocity and the velocity dispersion in km/s.
#     - Unless a good initial guess is available, it is recommended to set the starting
#       sigma >= 3*velScale in km/s (i.e. 3 pixels). In fact when the LOSVD is severely
#       undersampled, and far from the true solution, the chi^2 of the fit becomes weakly
#       sensitive to small variations in sigma (see pPXF paper). In some instances the
#       near-constancy of chi^2 may cause premature convergence of the optimization.
#     - In the case of two-sided fitting a good starting value for the
#       velocity is velStart=0.0 (in this case VSYST will generally be nonzero).
#       Alternatively on should keep in mind that velStart refers to the first
#       input galaxy spectrum, while the second will have velocity -velStart.
#
# KEYWORDS:
#   BESTFIT: a named variable to receive a vector with the best fitting
#       template: this is a linear combination of the templates, convolved with
#       the best fitting LOSVD, with added polynomial continuum terms.
#   BIAS: This parameter biases the (h3,h4,...) measurements towards zero
#       (Gaussian LOSVD) unless their inclusion significantly decreses the
#       error in the fit. Set this to BIAS=0.0 not to bias the fit: the
#       solution (including [V,sigma]) will be noisier in that case. The
#       default BIAS should provide acceptable results in most cases, but it
#       would be safe to test it with Monte Carlo simulations. This keyword
#       precisely corresponds to the parameter \lambda in the Cappellari &
#       Emsellem (2004) paper. Note that the penalty depends on the *relative*
#       change of the fit residuals, so it is insensitive to proper scaling
#       of the NOISE vector. A nonzero BIAS can be safely used even without a
#       reliable NOISE spectrum, or with equal weighting for all pixels.
#   /CLEAN: set this keyword to use the iterative sigma clipping method
#       described in Section 2.1 of Cappellari et al. (2002, ApJ, 578, 787).
#       This is useful to remove from the fit unmasked bad pixels, residual
#       gas emissions or cosmic rays.
#     - IMPORTANT: This is recommended *only* if a reliable estimate of the
#       NOISE spectrum is available. See also note below for SOL.
#   DEGREE: degree of the *additive* Legendre polynomial used to correct
#       the template continuum shape during the fit (default: 4).
#       Set DEGREE = -1 not to include any additive polynomial.
#   ERROR: a named variable that will contain a vector of *formal* errors
#       (1*sigma) for the fitted parameters in the output vector SOL. This 
#       option can be used when speed is essential, to obtain an order of 
#       magnitude estimate of the uncertainties, but we *strongly* recommend to 
#       run Monte Carlo simulations to obtain more reliable errors. In fact these 
#       errors can be severely underestimated in the region where the penalty 
#       effect is most important (sigma < 2*velScale).
#     - These errors are meaningless unless Chi^2/DOF~1 (see parameter SOL below).
#       However if one *assume* that the fit is good, a corrected estimate of the
#       errors is: errorCorr = error*sqrt(chi^2/DOF) = error*sqrt(sol[6]).
#     - IMPORTANT: when running Monte Carlo simulations to determine the error,
#       the penalty (BIAS) should be set to zero, or better to a very small value.
#       See Section 3.4 of Cappellari & Emsellem (2004) for an explanation.
#   GOODPIXELS: integer vector containing the indices of the good pixels in the
#       GALAXY spectrum (in increasing order). Only these pixels are included in
#       the fit. If the /CLEAN keyword is set, in output this vector will be updated
#       to contain the indices of the pixels that were actually used in the fit.
#     - IMPORTANT: in all likely situations this keyword *has* to be specified.
#   LAMBDA: When the keyword REDDENING is used, the user has to pass in this
#       keyword a vector with the same dimensions of GALAXY, giving the restframe
#       wavelength in Angstrom of every pixel in the input galaxy spectrum.
#       If one uses my LOG_REBIN routine to rebin the spectrum before the PPXF fit:
#           LOG_REBIN, lamRange, galaxy, galaxyNew, logLam
#       the wavelength can be obtained as lambda = EXP(logLam).
#   MDEGREE: degree of the *multiplicative* Legendre polynomial (with mean of 1)
#       used to correct the continuum shape during the fit (default: 0). The
#       zero degree multiplicative polynomial is always included in the fit as
#       it corresponds to the weights assigned to the templates.
#       Note that the computation time is longer with multiplicative
#       polynomials than with the same number of additive polynomials.
#     - IMPORTANT: Multiplicative polynomials cannot be used when
#       the REDDENING keyword is set.
#   MOMENTS: Order of the Gauss-Hermite moments to fit. Set this keyword to 4
#       to fit [h3,h4] and to 6 to fit [h3,h4,h5,h6]. Note that in all cases
#       the G-H moments are fitted (nonlinearly) *together* with [V,sigma].
#     - If MOMENTS=2 or MOMENTS is not set then only [V,sigma] are
#       fitted and the other parameters are returned as zero.
#     - If MOMENTS=0 then only the templates and the continuum additive
#       polynomials are fitted and the WEIGHTS are returned in output.
#   /OVERSAMPLE: Set this keyword to oversample the template by a factor 30x
#       before convolving it with a well sampled LOSVD. This can be useful to
#       extract proper velocities, even when sigma < 0.7*velScale and the 
#       dispersion information becomes totally unreliable due to undersampling. 
#       IMPORTANT: One should sample the spectrum more finely is possible, 
#       before resorting to the use of this keyword! 
#   /PLOT: set this keyword to plot the best fitting solution and the residuals
#       at the end of the fit.
#   POLYWEIGHTS: vector with the weights of the additive Legendre polynomials.
#       The best fitting additive polynomial can be explicitly evaluated as
#           x = cap_range(-1d,1d,n_elements(galaxy))
#           apoly = 0d # Additive polynomial
#           for j=0,DEGREE do apoly += legendre(x,j)*polyWeights[j]
#     - When doing a two-sided fitting (see help for GALAXY parameter), the additive
#       polynomials are allowed to be different for the left and right spectrum.
#       In that case the output weights of the additive polynomials alternate between
#       the first (left) spectrum and the second (right) spectrum.
#   /QUIET: set this keyword to suppress verbose output of the best fitting
#       parameters at the end of the fit.
#   REDDENING: Set this keyword to an initail estimate of the reddening E(B-V)>=0
#       to fit a positive reddening together with the kinematics and the templates.
#       After the fit the input estimate is replaced with the best fitting E(B-V) value.
#       The fit assumes the exctinction curve of Calzetti et al. (2000, ApJ, 533, 682)
#       but any other prescriptions could be trivially implemented by modifying the
#       function PPXF_REDDENING_CURVE below.
#     - IMPORTANT: The MDEGREE keyword cannot be used when REDDENING is set.
#   REGUL: If this keyword is nonzero, the program applies second-degree
#       linear regularization to the WEIGHTS during the PPXF fit.
#       Regularization is done in one, two or three dimensions depending on whether
#       the array of TEMPLATES has two, three or four dimensions respectively.
#       Large REGUL values correspond to smoother WEIGHTS output. The WEIGHTS tend
#       to a linear trend for large REGUL. When this keyword is nonzero the solution
#       will be a trade-off between smoothness of WEIGHTS and goodness of fit.
#     - The effect of the regularization scheme is to enforce the numerical second 
#       derivatives between neighbouring weights (in every dimension) to be equal 
#       to -w[j-1]+2*w[j]-w[j+1]=0 with an error Delta=1/REGUL. It may be helpful 
#       to define REGUL=1/Delta and view Delta as the regularization error.
#     - IMPORTANT: Delta needs to be smaller but of the same order of magnitude 
#       of the typical WEIGHTS to play an effect on the regularization. 
#       One way to achieve this is: 
#           (i) Divide the TEMPLATES array by a scalar in such a way that the typical 
#               template has a median of one (e.g. TEMPLATES/=median(TEMPLATES))# 
#          (ii) Do the same for the input GALAXY spectrum (e.g. GALAXY/=median(GALAXY)). 
#               In this situation a sensible guess for Delta will be a few percent.  
#     - Here is a possible recipe for chosing the regularization parameter REGUL:
#          (i) Perform an un-regularized fit (REGUL=0) and then rescale the input
#              NOISE spectrum so that Chi^2/DOF = Chi^2/N_ELEMENTS(goodPixels) = 1.
#              This is achieved by rescaling the input NOISE spectrum as
#              NOISE = NOISE*sqrt(Chi^2/DOF) = NOISE*sqrt(SOL[6])#
#         (ii) Increase REGUL and iteratively redo the pPXF fit until the Chi^2
#              increases from the unregularized Chi^2 = N_ELEMENTS(goodPixels)
#              value by DeltaChi^2 = sqrt(2*n_elements(goodPixels)).
#       The derived regularization corresponds to the maximum one still consistent
#       with the observations and the derived star formation history will be the
#       smoothest (minimum curvature) that is still consistent with the observations.
#     - For a detailed explanation see Section 18.5 of Press et al. (1992,
#       Numerical Recipes 2nd ed.) available here http://www.nrbook.com/a/bookfpdf.php.
#       The adopted implementation corresponds to their equation (18.5.10).
#   SKY: vector containing the spectrum of the sky to be included in the fit, or array
#       of dimensions SKY[nPixels,nSky] containing different sky spectra to add to
#       the model of the observed GALAXY spectrum. The SKY has to be log-rebinned as
#       the GALAXY spectrum and needs to have the same number of pixels.
#     - The sky is generally subtracted from the data before the PPXF fit. However,
#       for oservations very heavily dominated by the sky spectrum, where a very
#       accurate sky subtraction is critical, it may be useful *not* to subtract
#       the sky from the spectrum, but to include it in the fit using this keyword.
#   VSYST: galaxy systemic velocity (zero by default). The input initial guess
#       and the output velocities are measured with respect to this velocity.
#       The value assigned to this keyword is *crucial* for the two-sided fitting.
#       In this case VSYST can be determined from a previous normal one-sided
#       fit to the galaxy velocity profile. After that initial fit, VSYST
#       can be defined as the measured velocity at the galaxy center.
#       More accurately VSYST is the value which has to be subtracted to obtain
#       a nearly anti-symmetric velocity profile at the two opposite sides of
#       the galaxy nucleus.
#     - IMPORTANT: this value is generally *different* from the systemic
#       velocity one can get from the literature. Do not try to use that!
#   WEIGHTS: a named variable to receive the value of the weights by which each
#       template was multiplied to best fit the galaxy spectrum. The optimal
#       template can be computed with an array-vector multiplication:
#           TEMP = TEMPLATES # WEIGHTS (in IDL syntax)
#     - When the SKY keyword is used WEIGHTS[0:nTemplates-1] contains the weights
#       for the templates, while WEIGHTS[nTemplates:*] gives the ones for the sky.
#       In that case the best fitting galaxy template and sky are given by:
#           TEMP = TEMPLATES # WEIGHTS[0:nTemplates-1]
#           BESTSKY = SKY # WEIGHTS[nTemplates:*]
#     - When doing a two-sided fitting (see help for GALAXY parameter) *together*
#       with the SKY keyword, the sky weights are allowed to be different for the
#       left and right spectrum. In that case the output sky weights alternate
#       between the first (left) spectrum and the second (right) spectrum.
#
# OUTPUT PARAMETER:
#   SOL: seven elements vector containing in output the values of
#       [Vel,Sigma,h3,h4,h5,h6,Chi^2/DOF] of the best fitting solution, where DOF
#       is the number of Degrees of Freedom (number of fitted spectral pixels).
#       Vel is the velocity, Sigma is the velocity dispersion, h3-h6 are the
#       Gauss-Hermite coefficients. The model parameter are fitted simultaneously.
#     - I hardcoded the following safety limits on the fitting parameters:
#         a) Vel is constrained to be +/-2000 km/s from the first input guess
#         b) velScale/10 < Sigma < 1000 km/s
#         c) -0.3 < [h3,h4,...] < 0.3 (limits are extreme value for real galaxies)
#     - In the case of two-sided LOSVD fitting the output values refer
#       to the first input galaxy spectrum, while the second spectrum will
#       have by construction kinematics parameters [-Vel,Sigma,-h3,h4,-h5,h6].
#       If VSYST is nonzero (as required for two-sided fitting), then the
#       output velocity is measured with respect to VSIST.
#     - IMPORTANT: if Chi^2/DOF is not ~1 it means that the errors are not
#       properly estimated, or that the template is bad and it is *not* safe
#       to set the /CLEAN keyword.
#     - When MDEGREE > 1 then SOL contains in output the 7+MDEGREE elements
#       [Vel,Sigma,h3,h4,h5,h6,Chi^2/DOF,cx1,cx2,...,cxn], where cx1,cx2,...,cxn
#       are the coefficients of the multiplicative Legendre polynomials
#       of order 1,2,...,n. The polynomial can be explicitly evaluated as:
#           x = cap_range(-1d,1d,n_elements(galaxy))
#           mpoly = 1d # Multiplicative polynomial
#           for j=1,MDEGREE do mpoly += legendre(x,j)*sol[6+j]
#
#--------------------------------
# IMPORTANT: Proper usage of pPXF
#--------------------------------
#
# The PPXF routine can give sensible quick results with the default BIAS
# parameter, however, like in any penalized/filtered/regularized method, the
# optimal amount of penalization generally depends on the problem under study.
#
# The general rule here is that the penalty should leave the line-of-sight
# velocity-distribution (LOSVD) virtually unaffected, when it is well
# sampled and the signal-to-noise ratio (S/N) is sufficiently high.
#
# EXAMPLE: If you expect an LOSVD with up to a high h4 ~ 0.2 and your
# adopted penalty (BIAS) biases the solution towards a much lower h4 ~ 0.1,
# even when the measured sigma > 3*velScale and the S/N is high, then you
# are *misusing* the pPXF method!
#
# THE RECIPE: The following is a simple practical recipe for a sensible
# determination of the penalty in pPXF:
#
# 1. Choose a minimum (S/N)_min level for your kinematics extraction and
#    spatially bin your data so that there are no spectra below (S/N)_min#
#
# 2. Perform a fit of your kinematics *without* penalty (PPXF keyword BIAS=0).
#    The solution will be noisy and may be affected by spurious solutions,
#    however this step will allow you to check the expected mean ranges in
#    the Gauss-Hermite parameters [h3,h4] for the galaxy under study#
#
# 3. Perform a Monte Carlo simulation of your spectra, following e.g. the
#    included ppxf_simulation_example.pro routine. Adopt as S/N in the simulation 
#    the chosen value (S/N)_min and as input [h3,h4] the maximum representative
#    values measured in the non-penalized pPXF fit of the previous step#
#
# 4. Choose as penalty (BIAS) the *largest* value such that, for sigma > 3*velScale,
#    the mean difference delta between the output [h3,h4] and the input [h3,h4]
#    is well within (e.g. delta~rms/3) the rms scatter of the simulated values
#    (see e.g. Fig.2 of Emsellem et al. 2004, MNRAS, 352, 721).
#
#--------------------------------
#
# REQUIRED ROUTINES:
#       MPFIT: https://code.google.com/p/astrolibpy/
#
# MODIFICATION HISTORY:
#   V1.0 -- Created by Michele Cappellari, Leiden, 10 October 2001.
#   V3.47 -- First released version. MC, Leiden, 8 December 2003
#   V3.5 -- Included /OVERSAMPLE option. MC, Leiden, 11 December 2003
#   V3.6 -- Added MDEGREE option for multiplicative polynomials.
#           Linear implementation: fast, works well in most cases, but
#           can fail in certain cases. MC, Leiden, 19 March 2004
#   V3.7 -- Revised implementation of MDEGREE option. Nonlinear implementation:
#           straightforward, robust, but slower. MC, Leiden, 23 March 2004
#   V3.71 -- Updated documentation. MC, Leiden, 31 March 2004
#   V3.72 -- Corrected program stop after fit when MOMENTS=2.
#           Bug was introduced in V3.7. MC, Leiden, 28 April 2004
#   V3.73 -- Corrected bug: keyword ERROR was returned in pixels
#           instead of km/s. Decreased lower limit on fitted dispersion.
#           Thanks to Igor V. Chilingarian. MC, Leiden, 7 August 2004
#   V4.0 -- Introduced optional two-sided fitting assuming a reflection-symmetric
#           LOSVD for two input spectra. MC, Vicenza, 16 August 2004
#   V4.1 -- Corrected implementation of two-sided fitting of the LOSVD.
#           Thanks to Stefan van Dongen for reporting problems.
#           MC, Leiden, 3 September 2004
#   V4.11 -- Increased maximum number of iterations ITMAX in BVLS.
#           Thanks to Jesus Falcon-Barroso for reporting problems.
#           Introduced error message when velocity shift is too big.
#           Corrected output when MOMENTS=0. MC, Leiden, 21 September 2004
#   V4.12 -- Handle special case where a single template without additive
#           polynomials is fitted to the galaxy. MC, Leiden, 11 November 2004
#   V4.13 -- Updated documentation. MC, Vicenza, 30 December 2004
#   V4.14 -- Make sure input NOISE is a positive vector. MC, Leiden, 12 January 2005
#   V4.15 -- Verify that GOODPIXELS is monotonic and does not contain duplicated values.
#           After feedback from Richard McDermid. MC, Leiden, 10 February 2005
#   V4.16 -- Print number of nonzero templates. Do not print outliers in /QUIET mode.
#           MC, Leiden, 20 January 2006
#   V4.17 -- Updated documentation with important note on penalty determination.
#           MC, Oxford, 6 October 2007
#   V4.2 -- Introduced optional fitting of SKY spectrum. Many thanks to
#           Anne-Marie Weijmans for testing. MC, Oxford, 15 March 2008
#   V4.21 -- Use LA_LEAST_SQUARES (IDL 5.6) instead of SVDC when fitting
#           a single template. Please let me know if you need to use PPXF
#           with an older IDL version. MC, Oxford, 17 May 2008
#   V4.22 -- Added keyword POLYWEIGHTS. MC, Windhoek, 3 July 2008
#   V4.23 -- Corrected error message for too big velocity shift.
#           MC, Oxford, 27 November 2008
#   V4.3 -- Introduced REGUL keyword to perform linear regularization of WEIGHTS
#           in one or two dimensions. MC, Oxford, 4 Mach 2009
#   V4.4 -- Introduced Calzetti et al. (2000) PPXF_REDDENING_CURVE function to
#           estimate the reddening from the fit. MC, Oxford, 18 September 2009
#   V4.5 -- Dramatic speed up in the convolution of long spectra.
#           MC, Oxford, 13 April 2010
#   V4.6 -- Important fix to /CLEAN procedure: bad pixels are now properly
#           updated during the 3sigma iterations. MC, Oxford, 12 April 2011
#   V4.61 -- Use Coyote Graphics (http://www.idlcoyote.com/) by David W. Fanning.
#           The required routines are now included in NASA IDL Astronomy Library.
#           MC, Oxford, 29 July 2011
#   V4.62 -- Included option for 3D regularization and updated documentation of
#           REGUL keyword. MC, Oxford, 17 October 2011
#   V4.63 -- Do not change TEMPLATES array in output when REGUL is nonzero.
#           From feedback of Richard McDermid. MC, Oxford 25 October 2011
#   V4.64 -- Increased oversampling factor to 30x, when the /OVERSAMPLE keyword
#           is used. Updated corresponding documentation. Thanks to Nora 
#           Lu"tzgendorf for test cases illustrating errors in the recovered 
#           velocity when the sigma is severely undersampled.
#           MC, Oxford, 9 December 2011
#   V4.65 -- Expanded documentation of REGUL keyword. MC, Oxford, 15 November 2012
#   V4.66 -- Uses CAP_RANGE to avoid potential naming conflicts. 
#           MC, Paranal, 8 November 2013
#   V5.0 -- Translated from IDL into Python and tested against the original version.
#           MC, Oxford, 6 December 2013
#   V5.01 -- Minor cleaning and corrections. MC, Oxford, 12 December 2013
#   V5.1 -- Allow for a different LOSVD for each template. 
#           Templates can be stellar or can be gas emission lines. 
#           A PPXF version adapted for multiple kinematic components existed for years.
#           It was updated in JAN/2012 for the paper by Johnston et al. (2013, MNRAS).
#           This version merges those changes with the public PPXF version, making 
#           sure that all previous PPXF options are still supported. 
#           MC, Oxford, 9 January 2014
#   V5.11 -- Fixed typo in the documentation of nnls_flags. 
#           MC, Dallas Airport, 9 February 2014
#-
#----------------------------------------------------------------------------

import numpy as np    
import matplotlib.pyplot as plt
from numpy import polynomial
from scipy import ndimage, optimize, signal, linalg
import mpfit # https://code.google.com/p/astrolibpy/

eps = np.MachAr().eps

#----------------------------------------------------------------------------

def nnls_flags(A,b,flags):
    """
    Solves min||A*x - b|| with 
    x[j] >= 0 for flags[j] == 1
    x[j] free for flags[j] == 0
    where A[m,n], b[m], x[n], flags[n]
    
    """    
    m, n = A.shape
    mask = flags == 0
    AA = np.hstack([A, -A[:,mask]])
    x, err = optimize.nnls(AA, b)
    x[mask] -= x[n:]
    return x[:n]
        
#----------------------------------------------------------------------------

def rebin(x,factor):
    """
    Rebin a one-dimensional vector by averaging 
    in groups of "factor" adjacent values
    
    """
    return np.mean(x.reshape(-1,factor),axis=1)

#----------------------------------------------------------------------------

def robust_sigma(y, zero=False):
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

def reddening_curve(lam, ebv):
    """
    Reddening curve of Calzetti et al. (2000, ApJ, 533, 682# here C+00).
    This is reliable between 0.12 and 2.2 micrometres.
    - LAMBDA is the restframe wavelength in Angstrom of each pixel in the
      input galaxy spectrum (1 Angstrom = 1d-4 micrometres)
    - EBV is the assumed E(B-V) colour excess to redden the spectrum.
      In output the vector FRAC gives the fraction by which the flux at each
      wavelength has to be multiplied, to model the dust reddening effect.
      
    """    
    k1 = lam*0
    lam = 1e4/lam # Convert Angstrom to micrometres and take 1/lambda
    rv = 4.05 # C+00 equation (5)
    
    # C+00 equation (3) but extrapolate for lam>2.2
    # C+00 equation (4) but extrapolate for lam<0.12
    k1 = np.where(lam >= 6300, 
                  rv + 2.659*(1.040*lam - 1.857),
                  rv + 2.659*(1.509*lam - 0.198*lam**2 + 0.011*lam**3 - 2.156))
    fact = 10**(-0.4*ebv*(k1.clip(0)))  # Calzetti+00 equation (2) with opposite sign
    
    return fact # The model spectrum has to be multiplied by this vector

#----------------------------------------------------------------------------

def _bvls_solve(A, b, npoly):

    # No need to enforce positivity constraints if fitting one single template:
    # use faster linear least-squares solution instead of NNLS.
    #
    m, n = A.shape
    if m == 1: # A is a vector, not an array
        soluz = A.dot(b)/A.dot(A)
    elif n == npoly + 1: # Fitting a single template
        soluz = linalg.lstsq(A,b)[0]
    else:               # Fitting multiple templates
        flags = np.ones(n) 
        flags[:npoly] = 0 # flag = 0 on Legendre polynomials
        soluz = nnls_flags(A, b, flags)
    
    return soluz

#----------------------------------------------------------------------------,.

class ppxf(object):
    
    def __init__(self, templates, galaxy, noise, velScale, start, 
         bias=None, clean=False, degree=4, goodpixels=None, mdegree=0, 
         moments=2, oversample=False, plot=False, quiet=False, sky=None, 
         vsyst=0, regul=0, lam=None, reddening=None, component=0, reg_dim=None):

        # Do extensive checking of possible input errors
        #
        self.galaxy = galaxy
        self.noise = noise
        self.bias = bias
        self.clean = clean
        self.degree = degree
        self.mdegree = mdegree
        self.oversample = oversample
        self.quiet = quiet
        self.sky = sky
        self.vsyst = vsyst
        self.regul = regul
        self.lam = lam
        self.reddening = reddening
        self.reg_dim = np.asarray(reg_dim)
                
        s1 = templates.shape
        if len(s1) == 1: # Single template
            self.star = templates[:,np.newaxis]
        elif len(s1) == 2: # array of templates
            self.star = templates
        elif len(s1) >= 3: # >1-dim regularization
            self.star = templates.reshape(s1[0],-1)
            s1 = self.star.shape

        component = np.atleast_1d(component)
        if component.size == 1 and len(s1) > 1: # component is a scalar
            self.component = np.zeros(s1[1]) # all templates have the same LOSVD
        else:
            self.component = component
                        
        if len(s1) > 1:
            if self.component.size != s1[1]:
                raise ValueError('There must be one kinematic COMPONENT per template')

        tmp = np.unique(component)
        self.ncomp = tmp.size
        if not np.array_equal(tmp, np.arange(self.ncomp)):
            raise ValueError('must be 0 < COMPONENT < NCOMP-1')

        if regul > 0 and reg_dim is None:
            if self.ncomp == 1:
                self.reg_dim = np.asarray(templates.shape[1:])
            else:
                raise ValueError('reg_dim must be specified')            

        moments = np.atleast_1d(moments)
        if moments.size == 1: # moments is a scalar
            self.moments = np.zeros(self.ncomp) + np.abs(moments) # all LOSVD have the same number of G-H moments
        else:
            self.moments = np.abs(moments)

        if not np.array_equal(tmp.size, self.moments.size):
            raise ValueError('MOMENTS must be an array of length NCOMP')
            
        if self.regul is None:
            self.regul = 0
            
        s2 = self.galaxy.shape
        s3 = self.noise.shape
        
        if self.sky is not None:
            s4 = self.sky.shape
            if s4[0] != s2[0]:
                raise ValueError('SKY must have the same size as GALAXY')
                
        if (len(s1) > 2 or len(s2) > 2 or len(s3) > 2):
            raise ValueError('Wrong input dimensions')
            
        if not np.equal(s2,s3):
            raise ValueError('GALAXY and NOISE must have the same size/type')
            
        if (s1[0] < s2[0]):
            raise ValueError('STAR length cannot be smaller than GALAXY')
            
        if reddening is not None:
            if not np.equal(self.lam.shape,s2):
                raise ValueError('LAMBDA and GALAXY must have the same size/type')
            if self.mdegree != 0:
                raise ValueError('MDEGREE cannot be used with REDDENING keyword')
        else: 
            self.lam = 0

        if not np.all(noise > 0):
            raise ValueError('NOISE must be a positive vector')
            
        self.degree = max(degree, -1)
        self.mdegree = max(mdegree, 0)
        
        if oversample:
            self.factor = 30
        else:
            self.factor = 1    
            
        if goodpixels is None:
            self.goodpixels = np.arange(s2[0])
        else:
            if np.any(np.diff(goodpixels) <= 0):
                raise ValueError('goodpixels is not monotonic or contains duplicated values')
            if goodpixels[0] < 0 or goodpixels[-1] > s2[0]-1:
                raise ValueError('goodpixels are outside the data range')
            self.goodpixels = goodpixels
                
        if bias is None:
            self.bias = 0.7*np.sqrt(500./np.size(self.goodpixels)) # pPXF paper pg.144 left
            
        for j in range(self.ncomp):
            if self.moments[j] not in [0,2,4,6]:
                raise ValueError('MOMENTS should be 0, 2, 4 or 6')            

        if self.ncomp == 1:
            start = [start]
        
        if len(start) != self.ncomp:
            raise ValueError('START must have one element per kinematic component')
            
        if len(s2) == 2: 
            self.goodpixels = np.array([self.goodpixels, s2[0] + self.goodpixels])  # two-sided fitting of LOSVD
            
        if self.vsyst == 0:
            if len(s2) == 2:
                raise ValueError('VSYST must be defined for two-sided fitting')
        else:
            self.vsyst = self.vsyst / velScale

        ngh = int(self.moments.sum())        
        npars = ngh + self.mdegree*len(s2)
        
        if self.reddening is not None:
            npars += 1
        
        # Explicitly specify the step for the numerical derivatives
        # in MPFIT routine and force safety limits on the fitting parameters.
        #            
        # Set [h3,h4,...] and mult. polynomials to zero as initial guess
        # and constrain -0.3 < [h3,h4,...] < 0.3
        #
        parinfo = [{'step':1e-3, 'limits':[-0.3,0.3], 'limited':[1,1],
                    'value':0., 'fixed':0} for j in range(npars)] 
        
        p = 0            
        for j in range(self.ncomp):
            start1 = np.asarray(start[j][:2])/velScale # Convert velocity scale to pixels
            parinfo[0+p]['value'] = start1[0]
            parinfo[0+p]['limits'] = start1[0] + np.array([-2e3,2e3])/velScale # +/-2000 km/s from first guess
            parinfo[0+p]['step'] = 1e-2
            parinfo[1+p]['value'] = start1[1] 
            parinfo[1+p]['limits'] = [0.1,1e3/velScale] # hard-coded velScale/10<sigma<1000 km/s
            parinfo[1+p]['step'] = 1e-2                
            if s1[0] <= 2*(abs(self.vsyst) + abs(start1[0]) + 5.*start1[1]):
                raise ValueError('Velocity shift too big: Adjust wavelength ranges of spectrum and templates')                    
            if moments[j] < 0: # negative moments --> keep LOSVD fixed
                for k in range(self.moments[j]):
                    parinfo[k+p]['fixed'] = 1
                    if k > 1:
                        parinfo[k+p]['value'] = start[j][k]
            p += int(self.moments[j])

        if self.mdegree > 0:
            for j in range(ngh, npars):
                parinfo[j]['limits'] = [-1.,1.] # force <100% corrections
        elif self.reddening is not None:
            parinfo[ngh]['value'] = self.reddening
            parinfo[ngh]['limits'] = [0.,10.] # force positive E(B-V) < 10 mag
    
        # Here the actual calculation starts.
        # If required, once the minimum is found, clean the pixels deviating
        # more than 3*sigma from the best fit and repeat the minimization
        # until the set of cleaned pixels does not change any more.
        #
        good = self.goodpixels.copy()
        self.parinfo = parinfo
        for j in range(4): # Do at most five cleaning iterations
            self.clean = False # No cleaning during chi2 optimization
            mp = mpfit.mpfit(self._fitfunc, parinfo=parinfo, quiet=1, ftol=1e-4)
            self.pfit = mp.params
            ncalls = mp.nfev
            if clean is False:
                break
            goodOld = self.goodpixels.copy()
            self.goodpixels = good.copy()
            self.clean = True  # Do cleaning during linear fit
            tmp = self._fitfunc(mp.params)
            if np.array_equal(goodOld, self.goodpixels):
                break
                            
        # Evaluate scatter at the bestfit (with BIAS=0)
        # and also get the output bestfit and weights.
        #
        self.bias = 0
        status, err = self._fitfunc(mp.params)
        self.chi2 = robust_sigma(err, zero=True)**2       # Robust computation of Chi**2/DOF.
        
        p = 0
        self.sol = []
        self.error = []
        for j in range(self.ncomp):
            mp.params[p:p+2] *= velScale # Bring velocity scale back to km/s
            self.sol.append(mp.params[p:self.moments[j]+p])
            mp.perror[p:p+2] *= velScale # Bring velocity scale back to km/s
            self.error.append(mp.perror[p:self.moments[j]+p])
            p += self.moments[j]
        if mdegree >= 1:
            self.mpolyweights = mp.params[p:]                    
        if self.reddening is not None:
            self.reddening = mp.params[self.moments] # Replace input with best fit
        if degree >= 0:
            self.polyweights = self.weights[:(self.degree+1)*len(s2)] # output weights for the additive polynomials
        self.weights = self.weights[(self.degree+1)*len(s2):] # output weights for the templates (or sky) only

        if self.quiet is False:
            print "Best Fit:       V     sigma        h3        h4        h5        h6"
            for j in range(self.ncomp): 
                print "comp.", j, "".join("%10.3g" % f for f in self.sol[j])
            print "chi2/DOF:", self.chi2
            print 'Function evaluations:', ncalls
            nw = self.weights.size
            if reddening is not None:
                print 'Reddening E(B-V): ', self.reddening
            print 'Nonzero Templates: ', sum(self.weights > 0), ' / ', nw
            if self.weights.size <= 20:
                print 'Templates weights:'
                print "".join("%8.3g" % f for f in self.weights)

        if self.ncomp ==1:
            self.sol = self.sol[0]
            self.error = self.error[0]
        # Plot final data-model comparison if required.
        #
        if plot is True:
            mn = np.min(self.bestfit[self.goodpixels])
            mx = np.max(self.bestfit[self.goodpixels])
            n = self.goodpixels.size
            resid = mn + self.galaxy - self.bestfit
            mn1 = min(resid[self.goodpixels])
            plt.xlabel("Pixels")
            plt.ylabel("Counts")
            plt.xlim(np.array([-0.02,1.02])*self.galaxy.size)
            plt.ylim([mn1,mx] + np.array([-0.05,0.05])*(mx-mn1))            
            plt.plot(self.galaxy, '-k')
            plt.plot(self.bestfit, '-r', linewidth=2)
            plt.plot(self.goodpixels, self.goodpixels*0+mn, ',g')
            plt.plot(self.goodpixels, resid[self.goodpixels], 'dg', ms=4)
            w = np.where(np.diff(self.goodpixels) > 1)[0]
            if w.size > 0:
                for j in range(w.size):
                    x = np.arange(self.goodpixels[w[j]], self.goodpixels[w[j]+1])
                    plt.plot(x, resid[x],'-b')
                w = np.hstack([0, w, w+1, n-1])  # Add first and last point
            else:
                w = np.array([0, n-1])
            for j in range(w.size):
                plt.plot(self.goodpixels[w[[j,j]]], [mn, self.bestfit[self.goodpixels[w[j]]]], '-g')
            
        return

    #------------------------------------------------------------------

    def _fitfunc(self, pars, fjac=None):
        nspec = len(self.galaxy.shape)
        npix = self.galaxy.shape[0]
        npars = pars.size - self.mdegree*nspec  # Parameters of the LOSVD only
        if self.lam != 0:
            npars = npars - 1 # Fitting reddening
        
        # pars = [vel,sigma,h3,h4,...,m1,m2,...]    # Velocities are in pixels
        #
        dx = 0
        p = 0
        for j in range(self.ncomp): # loop over kinematic components
            tmp = np.ceil(abs(self.vsyst) + abs(pars[0+p]) + 5.*pars[1+p]) # Sample the Gaussian and GH at least to vsyst+vel+5*sigma
            dx = max(dx, tmp)
            p += self.moments[j]

        n = 2*dx*self.factor + 1
        x = np.linspace(-dx,dx,n)   # Evaluate the Gaussian using steps of 1/factor pixel
        losvd = np.empty((n,self.ncomp,nspec))
        p = 0
        for j in range(self.ncomp): # loop over kinematic components
            for k in range(nspec):    # nspec=2 for two-sided fitting, otherwise nspec=1
                if k ==0:
                    s = 1   # s=+1 for left spectrum, s=-1 for right one
                else:
                    s = -1
                vel = self.vsyst + s*pars[0+p]
                w = (x - vel)/pars[1+p]
                w2 = w**2
                losvd[:,j,k] = np.exp(-0.5*w2)/(np.sqrt(2.*np.pi)*pars[1+p]) # Normalized total(Gaussian)=1
            
                # Hermite polynomials normalized as in Appendix A of van der Marel & Franx (1993).
                # Coefficients for h5, h6 are given e.g. in Appendix C of Cappellari et al. (2002)
                #
                if self.moments[j] > 2:        # h_3 h_4
                    poly = 1 + s*pars[2+p]/np.sqrt(3)*(w*(2*w2-3)) \
                             + pars[3+p]/np.sqrt(24)*(w2*(4*w2-12)+3)
                    if self.moments[j] == 6:  # h_5 h_6
                        poly = poly + s*pars[4+p]/np.sqrt(60)*(w*(w2*(4*w2-20)+15)) \
                                    + pars[5+p]/np.sqrt(720)*(w2*(w2*(8*w2-60)+90)-15) 
                    losvd[:,j,k] = losvd[:,j,k]*poly
                p += self.moments[j]
        
        # The zeroth order multiplicative term is already included in the
        # linear fit of the templates. The polynomial below has mean of 1.
        #
        x = np.linspace(-1,1,npix) # X needs to be within [-1,1] for Legendre Polynomials
        mpoly = 1  # The loop below can be null if mdegree < 1
        if self.mdegree > 0:
            if nspec == 2: # Different multiplicative poly for left and right spectra
                mpoly1 = polynomial.legendre.legval( x, np.hstack([1.0, pars[npars::2]]) ) 
                mpoly2 = polynomial.legendre.legval( x, np.hstack([1.0, pars[npars+1::2]]) )
                mpoly = np.hstack([mpoly1, mpoly2])
            else: 
                mpoly = polynomial.legendre.legval( x, np.hstack([1.0, pars[npars:]]) ) 
        
        # Multiplicative polynomials do not make sense when fitting reddening.
        # In that case one has to assume the spectrum is well calibrated.
        #
        if self.lam != 0: 
            mpoly = reddening_curve(self.lam, pars[npars])
        
        # Fill the columns of the design matrix of the least-squares problem
        #
        skydim = len(np.shape(self.sky))
        if skydim == 0:
            nsky = 0
        elif skydim == 1:
            nsky = 1 # Number of sky spectra
        else:
            nsky = np.shape(self.sky)[1]
    
        tempdim = len(self.star.shape)
        if tempdim == 2:
            ntemp = np.shape(self.star)[1]    # Number of template spectra
        else:
            ntemp = 1
        
        nrows = (self.degree + 1 + nsky)*nspec + ntemp
        ncols = npix*nspec
        if self.regul > 0:
            dim = self.reg_dim.size
            reg2 = self.reg_dim - 2
            if dim == 1:
                nreg = reg2
            elif dim == 2:
                nreg = 2*np.prod(reg2) + 2*np.sum(reg2) # Rectangle sides have one finite difference
            elif dim == 3:                               # Hyper-rectangle edges have one finite difference
                nreg = 3*np.prod(reg2) + 4*np.sum(reg2) 
                + 4*( np.prod(reg2[[0,1]]) + np.prod(reg2[[0,2]]) + np.prod(reg2[[1,2]]) )
            ncols = ncols + nreg
        c = np.zeros((ncols,nrows))  # This array is used for estimating predictions
        
        for j in range(self.degree+1): # Fill first columns of the Design Matrix
            coeff = np.zeros(j+1)
            coeff[j] = 1.0
            leg = polynomial.legendre.legval(x, coeff)
            if nspec == 2:
                c[:npix*nspec,2*j] = np.hstack([leg,leg*0])   # Additive polynomials for left spectrum
                c[:npix*nspec,2*j+1] = np.hstack([leg*0,leg]) # Additive polynomials for right spectrum
            else: 
                c[:npix,j] = leg

        # Accounts for difference between IDL's CONVOL and signal.fftconvolve
        # CONVOLVE(a,b,/EDGE_ZERO) = signal.fftconvolve(a,b[::-1],'same')
        
        tmp = np.empty((self.star.shape[0],nspec))
        for j in range(ntemp):
            if self.factor == 1: # No oversampling of the template spectrum
                for k in range(nspec):
                    tmp[:,k] = signal.fftconvolve(self.star[:,j],losvd[:,self.component[j],k],mode='same')
            else:             # Oversample the template spectrum before convolution
                st = ndimage.interpolation.zoom(self.star[:,j],self.factor,order=1)
                for k in range(nspec):
                    tmp[:,k] = rebin(signal.fftconvolve(st,losvd[:,self.component[j],k],mode='same'),self.factor)
            c[:npix*nspec,(self.degree+1)*nspec+j] = mpoly*tmp[:npix,:].ravel() # reform into a vector
        
        # Add second-degree 1D, 2D or 3D linear regularization
        # Press W.H., et al., 1992, Numerical Recipes, 2nd ed. equation (18.5.10)
        #
        if self.regul > 0:
            if dim == 1:
                i = np.arange(self.reg_dim)
            else:
                i = np.arange(np.prod(self.reg_dim)).reshape(self.reg_dim)
            i += (self.degree+1)*nspec
            p = npix*nspec
            diff = np.array([-1,2,-1])*self.regul
            ind = np.array([-1,0,1])
            if dim == 1:
                for j in range(1,self.reg_dim-1): 
                    c[p,i[j+ind]] = diff
                    p += 1
            elif dim == 2:
                for k in range(self.reg_dim[1]):
                    for j in range(self.reg_dim[0]):
                        if j != 0 and j != self.reg_dim[0]-1:
                            c[p,i[j+ind,k]] = diff
                            p += 1
                        if k != 0 and k != self.reg_dim[1]-1:
                            c[p,i[j,k+ind]] = diff
                            p += 1
            elif dim == 3:
                for q in range(self.reg_dim[2]):
                    for k in range(self.reg_dim[1]):
                        for j in range(self.reg_dim[0]):
                            if j != 0 and j != self.reg_dim[0]-1:
                                c[p,i[j+ind,k,q]] = diff
                                p += 1
                            if k != 0 and k != self.reg_dim[1]-1:
                                c[p,i[j,k+ind,q]] = diff
                                p += 1
                            if q != 0 and q != self.reg_dim[2]-1:
                                c[p,i[j,k,q+ind]] = diff
                                p += 1
        
        for j in range(nsky):
            skyj = self.sky[:,j]
            k = (self.degree+1)*nspec + ntemp
            if nspec == 2:
                c[:npix*nspec,k+2*j] = [skyj,skyj*0]   # Sky for left spectrum
                c[:npix*nspec,k+2*j+1] = [skyj*0,skyj] # Sky for right spectrum
            else: 
                c[:npix,k+j] = skyj

        npoly = (self.degree+1)*nspec # Number of additive polynomials in the fit
        
        # Select the spectral region to fit and solve the overconditioned system
        # using SVD/BVLS. Use unweighted array for estimating bestfit predictions.
        # Iterate to exclude pixels deviating more than 3*sigma if /CLEAN keyword is set.
        #
        a = c.copy() # This array is used for the actual solution of the system
        a[:npix*nspec,:] = c[:npix*nspec,:] / self.noise[:,np.newaxis] # Weight all columns with errors
        b = self.galaxy / self.noise
                
        m = 1
        while m != 0:
            if self.regul > 0:
                aa = a[np.hstack([self.goodpixels, np.arange(npix*nspec,ncols)]), :]
                bb = np.hstack([b[self.goodpixels], np.zeros(nreg)])
            else:
                aa = a[self.goodpixels,:]
                bb = b[self.goodpixels]
            self.weights = _bvls_solve(aa,bb,npoly)
            self.matrix = c[:npix*nspec,:]
            self.bestfit = c[:npix*nspec,:].dot(self.weights)
            err = (self.galaxy[self.goodpixels] - self.bestfit[self.goodpixels]) \
                /  self.noise[self.goodpixels]
            if self.clean is True:
                w = np.where(np.abs(err) < 3) # select residuals smaller than 3*sigma
                m = np.size(err) - np.size(w)
                if m != 0:
                    self.goodpixels = self.goodpixels[w]
                    if self.quiet != 1: 
                        print 'Outliers:', m
            else: 
                break
        
        # Penalize the solution towards (h3,h4,...)=0 if the inclusion of
        # these additional terms does not significantly decrease the error.
        #
        if np.any(self.moments > 2) and self.bias != 0:
            p = 0
            tmp = 0.
            for j in range(self.ncomp): # loop over kinematic components
                if self.moments[j] > 2:
                    tmp += np.sum(pars[2+p:self.moments[j]+p]**2)
                p += self.moments[j]
            err += self.bias*robust_sigma(err, zero=True)*np.sqrt(tmp)

        status = 0
        
        return (status, err)

#----------------------------------------------------------------------------
