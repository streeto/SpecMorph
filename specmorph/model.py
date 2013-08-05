'''
Created on Jun 6, 2013

@author: andre
'''

import numpy as np
from scipy import ndimage

from astropy.modeling import Parametric1DModel

__all__ = ['BulgeModel', 'DiskModel', 'GalaxyModel', 'PSF1d_gaussian_convolve']

################################################################################
def disk_profile(r, I_D0, R_0):
    if R_0 <= 0.0 or I_D0 <= 0.0:
        return np.zeros_like(r)
    return I_D0 * np.exp(-r / R_0)


def bulge_profile(r, I_Be, R_e):
    if R_e <= 0.0 or I_Be <= 0.0:
        return np.zeros_like(r)
    return I_Be * np.exp(-7.669 * ((r / R_e)**0.25 - 1))



def PSF1d_gaussian_convolve(sigma, r, func):
    if sigma == 0.0:
        return func(r)
    
    subpix_resolution = 10.0
    r_min = r.min()
    _r = np.arange(r_min, r.max(), 1/subpix_resolution)

    _I = func(_r)
    _I = ndimage.gaussian_filter1d(_I, subpix_resolution*sigma, mode='reflect')
    return np.interp(r, _r, _I)
################################################################################


################################################################################
class BulgeModel(Parametric1DModel):
    param_names = ['I_Be', 'R_e']

    def __init__(self, I_Be, R_e, **cons):
        super(BulgeModel, self).__init__(locals(), **cons)
        self.deriv = None


    def eval(self, r, I_Be, R_e):
        return bulge_profile(r, I_Be, R_e)
################################################################################

    
################################################################################
class DiskModel(Parametric1DModel):
    param_names = ['I_D0', 'R_0']

    def __init__(self, I_D0, R_0, **cons):
        super(DiskModel, self).__init__(locals(), **cons)
        self.deriv = None


    def eval(self, r, I_D0, R_0):
        return disk_profile(r, I_D0, R_0)
################################################################################



################################################################################
class GalaxyModel(Parametric1DModel):
    param_names = ['I_Be', 'R_e', 'I_D0', 'R_0']
    R2 = None


    def __init__(self, I_Be, R_e, I_D0, R_0, sigma, **cons):
        super(GalaxyModel, self).__init__(locals(), **cons)
        self._sigma = sigma
        self.linear = False
        self.deriv = None


    def eval(self, r, I_Be, R_e, I_D0, R_0):
        def bulge_disk_profile(r):
            I_B = bulge_profile(r, I_Be, R_e)
            I_D = disk_profile(r, I_D0, R_0)
            return I_B + I_D
        return PSF1d_gaussian_convolve(self._sigma, r, bulge_disk_profile)


    def unbound_deriv(self, r, I_Be, R_e, I_D0, R_0):
        I_B = bulge_profile(r, I_Be, R_e)
        I_D = disk_profile(r, I_D0, R_0)
        d_I_Be = I_B / I_Be
        d_Re = I_B * (7.669 / 4.0 / R_e**1.25) * r**0.25
        d_I_D0 = I_D / I_D0
        d_R0 = I_D / R_0**2 * r
        return [d_I_Be, d_Re, d_I_D0, d_R0]

        
    def bound_deriv(self, r, I_Be, R_e, I_D0, R_0):
        d_I_Be, d_Re, d_I_D0, d_R0 = self.unbound_deriv(r, I_Be, R_e, I_D0, R_0).T
        d_I_Be = self._bound_deriv_param(self._I_Be, I_Be, d_I_Be)
        d_Re = self._bound_deriv_param(self._R_e, R_e, d_Re)
        d_I_D0 = self._bound_deriv_param(self._I_D0, I_D0, d_I_D0)
        d_R0 = self._bound_deriv_param(self._R_0, R_0, d_R0)
        return [d_I_Be, d_Re, d_I_D0, d_R0]


    def _bound_deriv_param(self, param, value, deriv):
        # FIXME: why these, you ask? No idea, it just fitted better this way.
        lo_threshold = 0.2
        hi_threshold = 0.8
        k = 1000.0
        vmin, vmax = self.constraints.bounds[param._name]
        v = (value - vmin) / (vmax - vmin)
        if v < lo_threshold:
            return deriv - k * (v - lo_threshold)**4
        elif v > hi_threshold:
            return deriv + k * (v - hi_threshold)**4
        return deriv
################################################################################



