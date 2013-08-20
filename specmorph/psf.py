'''
Created on Aug 20, 2013

@author: andre
'''

import numpy as np
from scipy import ndimage


################################################################################
def PSF1d_gaussian_convolve(sigma, r, func):
    if sigma == 0.0:
        return func(r)
    
    conv_samples = 100.0
    r_scale = r.max()
    dr = r_scale / conv_samples
    _r = np.arange(r.min(), r_scale + dr, dr)
    _I = ndimage.gaussian_filter1d(func(_r), sigma / r.max() * conv_samples, mode='mirror')
    return np.interp(r, _r, _I)
################################################################################


################################################################################
def gaussian2d_kernel(sigma):
    center = int(4.0 * sigma)
    fw = 2 * center + 1
    xx, yy = np.indices((fw, fw), dtype='int')
    rr2 = (xx - center)**2 + (yy - center)**2
    return np.exp(- 0.5 * rr2 / sigma**2) / (2.0 * np.pi * sigma**2)
################################################################################
