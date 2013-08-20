'''
Created on Jun 6, 2013

@author: andre
'''

import numpy as np

from astropy.modeling import Parametric1DModel
from .psf import PSF1d_gaussian_convolve

__all__ = ['BulgeModel', 'DiskModel', 'GalaxyModel', 'PSF1d_gaussian_convolve']

################################################################################
def disk_profile(r, I_D0, R_0):
    return I_D0 * np.exp(-r / R_0)


def bulge_profile(r, I_Be, R_e):
    return I_Be * np.exp(-7.669 * ((r / R_e)**0.25 - 1))
################################################################################


################################################################################
class GalaxyModel(Parametric1DModel):
    param_names = ['I_Be', 'R_e', 'I_D0', 'R_0']
    R2 = None


    def __init__(self, I_Be, R_e, I_D0, R_0, sigma, **constraints):
        super(GalaxyModel, self).__init__(locals())
        self._sigma = sigma
        self.linear = False


    def eval(self, r, I_Be, R_e, I_D0, R_0):
        def bulge_disk_profile(r):
            I_B = bulge_profile(r, I_Be, R_e)
            I_D = disk_profile(r, I_D0, R_0)
            return I_B + I_D
        return PSF1d_gaussian_convolve(self._sigma, r, bulge_disk_profile)


    def deriv(self, r, I_Be, R_e, I_D0, R_0):
        def dI_Be(r):
            return bulge_profile(r, I_Be, R_e) / I_Be
        
        def dRe(r):
            return bulge_profile(r, I_Be, R_e) * (7.669 / 4.0 / R_e**1.25) * r**0.25
        
        def dI_D0(r):
            return disk_profile(r, I_D0, R_0) / I_D0

        def dR0(r):
            return  disk_profile(r, I_D0, R_0) / R_0**2 * r
        
        return [PSF1d_gaussian_convolve(self._sigma, r, dI_Be),
                PSF1d_gaussian_convolve(self._sigma, r, dRe),
                PSF1d_gaussian_convolve(self._sigma, r, dI_D0),
                PSF1d_gaussian_convolve(self._sigma, r, dR0)]

################################################################################


################################################################################
class BulgeModel(GalaxyModel):

    def __init__(self, I_Be, R_e, sigma=0.0, **constraints):
        super(BulgeModel, self).__init__(I_Be=I_Be, R_e=R_e,
                                         I_D0=0.0, R_0=1.0,
                                         sigma=sigma, **constraints)
################################################################################

    
################################################################################
class DiskModel(GalaxyModel):

    def __init__(self, I_D0, R_0, sigma=0.0, **constraints):
        super(DiskModel, self).__init__(I_Be=0.0, R_e=1.0,
                                        I_D0=I_D0, R_0=R_0,
                                        sigma=sigma, **constraints)
################################################################################



