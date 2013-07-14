'''
Created on Jun 6, 2013

@author: andre
'''

import numpy as np
from scipy import ndimage

from astropy.modeling import ParametricModel, Parameter
from astropy.modeling.core import _convert_input, _convert_output

__all__ = ['BulgeModel2D', 'DiskModel2D', 'GalaxyModel']

################################################################################
def disk_profile(r, I_D0, R_0):
    return I_D0 * np.exp(-r / R_0)


def bulge_profile(r, I_Be, R_e):
    return I_Be * np.exp(-7.669 * ((r / R_e)**0.25 - 1))


def test_profile(r):
    return 1.0 * np.exp(-7.669 * ((r / 5.0)**0.25 - 1))

def PSF_gaussian_convolve(sigma, r, func):
    if sigma == 0.0:
        return func(r)
    _r = np.arange(0.1, r.max())
    _I = func(_r)
    _I = ndimage.gaussian_filter(_I, sigma, mode='reflect')
    return np.interp(r, _r, _I)
################################################################################


################################################################################
class BulgeModel2D(ParametricModel):
    param_names = ['I_Be', 'R_e']

    def __init__(self, I_Be, R_e, sigma, param_dim=1):
        self._I_Be = Parameter(name='I_Be', val=I_Be, mclass=self, param_dim=param_dim)
        self._R_e = Parameter(name='R_e', val=R_e, mclass=self, param_dim=param_dim)
        self._sigma = sigma
        ParametricModel.__init__(self, self.param_names, n_inputs=1, n_outputs=1, param_dim=param_dim)
        self.linear = False
        self.deriv = None


    def eval(self, r, params):
        I_Be, R_e = params
        I_B = bulge_profile(r, I_Be, R_e)
        return ndimage.gaussian_filter(I_B, self._sigma)

                                  
    def __call__(self, r):
        r, fmt = _convert_input(r, self.param_dim)
        result = self.eval(r, self.param_sets)
        return _convert_output(result, fmt)
################################################################################

    
################################################################################
class DiskModel2D(ParametricModel):
    param_names = ['I_D0', 'R_0']

    def __init__(self, I_D0, R_0, sigma, param_dim=1):
        self._I_D0 = Parameter(name='I_D0', val=I_D0, mclass=self, param_dim=param_dim)
        self._R_0 = Parameter(name='R_0', val=R_0, mclass=self, param_dim=param_dim)
        self._sigma = sigma
        ParametricModel.__init__(self, self.param_names, n_inputs=1, n_outputs=1, param_dim=param_dim)
        self.linear = False
        self.deriv = None


    def eval(self, r, params):
        I_D0, R_0 = params
        I_D = disk_profile(r, I_D0, R_0)
        return ndimage.gaussian_filter(I_D, self._sigma)


    def __call__(self, r):
        r, fmt = _convert_input(r, self.param_dim)
        result = self.eval(r, self.param_sets)
        return _convert_output(result, fmt)
################################################################################



################################################################################
class GalaxyModel(ParametricModel):
    param_names = ['I_Be', 'R_e', 'I_D0', 'R_0']
    R2 = None


    def __init__(self, I_Be, R_e, I_D0, R_0, sigma, **cons):
        try:
            param_dim = len(I_Be)
        except TypeError:
            param_dim = 1
        self._I_Be = Parameter(name='I_Be', val=I_Be, mclass=self, param_dim=param_dim)
        self._R_e = Parameter(name='R_e', val=R_e, mclass=self, param_dim=param_dim)
        self._I_D0 = Parameter(name='I_D0', val=I_D0, mclass=self, param_dim=param_dim)
        self._R_0 = Parameter(name='R_0', val=R_0, mclass=self, param_dim=param_dim)
        self._sigma = sigma
        ParametricModel.__init__(self, self.param_names, n_inputs=1, n_outputs=1, param_dim=param_dim, **cons)
        self.linear = False
        self.deriv = self.unbound_deriv


    def eval(self, r, params):
        I_Be, R_e, I_D0, R_0 = params
        def bulge_disk_profile(r):
            I_B = bulge_profile(r, I_Be, R_e)
            I_D = disk_profile(r, I_D0, R_0)
            return I_B + I_D
        return PSF_gaussian_convolve(self._sigma, r, bulge_disk_profile)


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
        lo_threshold = 0.2
        hi_threshold = 0.8
        k = 1000.0
        vmin, vmax = self.constraints.bounds[param._name]
        v = value - vmin / (vmax - vmin)
        if v < lo_threshold:
            return deriv + k * (v - lo_threshold)**4
        elif v > hi_threshold:
            return deriv + k * (v - hi_threshold)**4
        return deriv


    def __call__(self, r):
        r, fmt = _convert_input(r, self.param_dim)
        result = self.eval(r, self.param_sets)
        return _convert_output(result, fmt)
################################################################################



