# -*- coding: utf-8 -*-
"""# Constants
c = 299792.458 # Speed of light
FWHM_tem = 2.54 # MILES library spectra have a resolution FWHM of 2.54A.
FWHM_spec = 2.1 # FORS2 for Hydra observations has an instrumental
               # resolution FWHM of 4.2A.

Created on 12/09/15

@author: Carlos Eduardo Barbosa

"""

# Path configurations
home = "/home/kadu/Dropbox/hydra1"
template_dir = home + "/templates"
data_dir = home + "/data/1d/"
tables_dir = home + "/tables"
images_dir = home + "/images"
figures_dir = home + "/figs"

# Set some global constants0
re = 8.4 # Effective radius in kpc, value from Arnaboldi
# re = 17.73 # Value from 2MASS multiplied by 1.7 as in Atlas3D Paper I
# re = 26.6 # Effective radius in kpc, value from Loubser & Sanchez-Blazquez 2012
sn_cut= 5. # Minimum S/N to be used
pa0 = 63. # Photometric position angle
velscale = 30. # Set velocity scale for pPXF related routines

# Constants
c = 299792.458 # Speed of light
FWHM_tem = 2.54 # MILES library spectra have a resolution FWHM of 2.54A.
FWHM_spec = 2.1 # FORS2 for Hydra observations has an instrumental
               # resolution FWHM of 4.2A.
