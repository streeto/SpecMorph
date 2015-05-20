'''
Created on 20/06/2014

@author: andre
'''

import logging
import numpy as np

__all__ = ['logger', 'find_nearest_index', 'gaussian_covariance']

logger = logging.getLogger('specmorph')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.ERROR)


def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def gaussian_covariance(noise, wl, wl_FWHM):
    nl = len(wl)
    dl = (wl[-1] - wl[0]) / nl
    theta = wl_FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    i = np.arange(nl)[:, np.newaxis]
    j = i.T
    A = dl / (np.sqrt(2.0 * np.pi) * theta)
    B = -0.5 * (dl / theta)**2
    p = A * np.exp(B * (i - j)**2)
    mask = np.identity(nl, 'bool')
    p[mask] = 0.0
    noise_t = np.transpose(noise, (1, 2, 0))
    cov = np.dot(noise_t, p)
    cov *= noise_t
    return cov.sum(axis=2)
    
    
