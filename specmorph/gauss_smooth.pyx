'''
Created on Jun 13, 2013

@author: andre
'''
from __future__ import division
import numpy as np
cimport numpy as np


cdef extern from 'math.h':
    double exp(double x)
    double sqrt(double x)


cdef inline double _interp(double fo_a, double fo_b, double lo_a, double lo_b, double lo_x):
                cdef double a, b, ff
                a = (fo_b - fo_a) / (lo_b - lo_a)
                b = fo_a - a * lo_a
                ff = a * lo_x + b
                return ff


cdef double c = 2.997925e5  # km/s
cdef sqrt_2pi = sqrt(2.0 * np.pi)

def gaussVelocitySmooth(np.ndarray lo not None, np.ndarray fo not None,
                        double v0, double sig, np.ndarray ls=None, int n_u=31, int n_sig=6):
    '''
    Apply a gaussian velocity dispersion and displacement filter to a spectrum.
    Implements the integration presented in the page 27 of the 
    `STARLIGHT manual <http://www.starlight.ufsc.br/papers/Manual_StCv04.pdf>`_.
    
    Parameters
    ----------
    
    lo : array
        Original wavelength array.
        
    fo : array
        Original flux array, must be of same length as ``lo``.
    
    v0 : float
        Systemic velocity to apply to input spectrum.
        
    sig : float
        Velocity dispersion (sigma of the gaussian).
        
    ls : array, optional
        Wavelengths in which calculate the output spectrum.
        If not set, ``lo`` will be used.
        
    n_u : int, optional
        Number of points to sample the velocity gaussian.
        Defaults to 31. Use an odd number to guarantee the center
        of the gaussian enters the calculation.
        
    n_sig : int, optional
        Width of the integration kernel, in units of sigma.
        
    Returns
    -------
    
    fs : array
        Resampled and smoothed flux array, must be of same length as ``ls``
        (or ``lo`` if ``ls`` is not set).
    
   '''




    if n_u < 5:
        raise ValueError('n_u=%d too small to integrate properly.' % n_u)

    if ls is None:
        ls = lo
        
# Parameters for the brute-force integration in velocity
    cdef double u_low = -n_sig
    cdef double u_upp = n_sig
    cdef double du = (u_upp - u_low) / (n_u - 1)

    cdef double d_lo = lo[1] - lo[0]

    cdef int Ns = ls.shape[0]
    cdef int No = lo.shape[0]
    
    cdef double sum_fg, u, v, ll, a, b, ff
    cdef double lo_0 = lo[0]
    cdef double fo_0 = fo[0]
    cdef double fo_last = fo[No - 1]

    cdef int i_s, ind
    cdef np.ndarray fs = np.empty(Ns, dtype=np.float64)

# loop over ls, convolving fo(lo) with a gaussian
    for i_s in xrange(Ns):

# reset integral of {fo[ll = ls/(1+v/c)] * Gaussian} & start integration
        sum_fg = 0.
        for i_u in xrange(n_u):
            u = u_low + i_u * du
# define velocity & lambda corresponding to u
            v = v0 + sig * u
            ll = ls[i_s] / (1.0 + v / c)

# find fo flux for lambda = ll
            ind = int((ll - lo_0) / d_lo)
            if ind < 0: 
                ff = fo_0
            elif ind >= No - 1:
                ff = fo_last
            else:
                ff = _interp(fo[ind + 1], fo[ind], lo[ind + 1], lo[ind], ll)

# smoothed spectrum
            sum_fg = sum_fg + ff * exp(-(u ** 2 / 2.)) 
        fs[i_s] = sum_fg * du / sqrt_2pi
    return fs


