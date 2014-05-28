'''
Created on Jul 22, 2013

@author: andre
'''

import numpy as np
from scipy.signal import detrend

__all__ = ['integrated_spec', 'flag_big_error', 'flag_small_error']

tsfe = 0.1
tbfe = 5.0
inf = 11
ibe = 13
ise = 9

def areafactor(area, M=15.0):
    '''
    Not sure what this does. Comment found in qbick:
    # M = 14.3 (para 30 galaxias)
    '''
    return np.sqrt(M * area / (M + area - 1))


def get_spec_sum(spec_arr, sper_arr, spef_arr, nz, beta, fg=2.0):
    '''
    Get spectrum error of the sum of zones.
    '''
    spec = np.ma.masked_where(spef_arr > 0.0, spec_arr)
    # Weight of each zone.
    w = np.ma.median(spec, axis=0)
    w /= w.sum()
    completeness = ((~spec.mask).astype('float64')*w).sum(axis=1)
    hasdata = completeness > 0.0
    # Sum of spectra, errors and flags
    spec_sum = spec.sum(axis=1)
    spec_sum[hasdata] /= completeness[hasdata]
    sper_sum2 = (np.power(sper_arr, 2.0) * beta**2).sum(axis=1)
    sper_sum2[hasdata] /= completeness[hasdata]
    sper_sum = np.sqrt(sper_sum2)
    spef_sum = np.where(spec_sum.mask, 1.0, 0.0)  # Total flag * Rule "botella medio llena"
    spef_sum[~hasdata] = 1.0
    return np.asarray(spec_sum), sper_sum, spef_sum


def integrated_spec(f_obs, f_err, f_flag):
    N_zone = f_obs.shape[1]
    beta__z = areafactor(N_zone)
    i_f_obs, i_f_err, i_f_flag = get_spec_sum(f_obs, f_err, f_flag, N_zone, beta__z)

    i_f_flag[np.where(i_f_flag > 0)] = inf
    fber_sum = np.where(i_f_err > tbfe * abs(i_f_obs), ibe, 0.0)
    sm_err = np.median(i_f_err[np.where((i_f_obs > 0.0) & (i_f_err > 0.0) & (fber_sum < 1.0) & (i_f_flag < 1.0))])
    fser_sum = np.where(i_f_err < tsfe * sm_err, ise, 0.0)  # Flag for small Errors
    i_f_flag = i_f_flag + fber_sum + fser_sum  # Total flag
    
    # Scale error to match the RMS of the flux.
    good = i_f_flag<1
    sig_rms = DerNoise1D(i_f_obs[good], masked=False)
    noise = np.median(i_f_err[good])
    #signal = np.median(i_f_obs[good])
    sfactor = (sig_rms/noise)
    
    # normalize to S/N = 50.
    #sfactor *= (signal / noise / 50.0)
    
    # Only do this of error is too small.
    if sfactor > 1.0:
        i_f_err *= sfactor
    
    return i_f_obs, i_f_err, i_f_flag


def flag_big_error(f_obs, f_err):
    return np.where(f_err > tbfe * abs(f_obs), ibe, 0.0)


def flag_small_error(f_obs, f_err, f_flag):
    m_err = np.median(f_err[np.where((f_obs > 0.0) & (f_err > 0.0) & (f_flag < 1.0))]) 
    return np.where(f_err < tsfe * m_err, ise, 0.0)


def calc_sn(l, f, flag):
    if isinstance(f, np.ma.MaskedArray):
        f = f.copy()
    else:
        f = np.ma.array(f, copy=True, fill_value=np.nan)
    f[(flag > 0)] = np.ma.masked

    signal = np.ma.median(f, axis=0)
    noise = np.ma.std(detrend(f, axis=0), axis=0)
    return signal, noise, signal / noise


def mdetrend(x,y,axis=0):
    if len(np.shape(x)) == 1 and len(np.shape(y)) == 1:
        #linreg = stats.mstats.linregress(x,y)
        linreg = np.ma.polyfit(x,y,1)
        # [pendiente, ordenada_origen] --> linreg[0]: slope; linreg[1]: intercept
        z = linreg[1] + linreg[0]*x
        return y-z
    elif len(np.shape(x)) == 1 and len(np.shape(y)) >= 2: 
        data = y.copy()
        dshape = data.shape
        dtype = data.dtype.char
        N = dshape[axis]
#         Npts = np.shape(data)[axis]
        # Restructure data so that axis is along first dimension and
        #  all other dimensions are collapsed into second dimension
        rnk = len(dshape)
        if axis < 0: axis = axis + rnk
        newdims = np.r_[axis,0:axis,axis+1:rnk]
        newdata = np.reshape(np.transpose(data, tuple(newdims)),(N, np.prod(dshape, axis=0)/N))
        newdata = newdata.copy()  # make sure we have a copy
        if newdata.dtype.char not in 'dfDF':
            newdata = newdata.astype(dtype)
        ns = np.shape(newdata)[1]
        # Polyfit RankWarning: The rank of the coefficient matrix in the least-squares could be deficient. 
        # We turn off the warnings at the beginning of the module
        coef = np.array([np.ma.polyfit(x,newdata[:,k],1) for k in range(ns)])
        dtrendat = newdata - (x[:,np.newaxis]*coef[:,0] + coef[:,1])
        tdshape = np.take(dshape,newdims,0)
        ret = np.reshape(dtrendat,tuple(tdshape))
        vals = range(1,rnk)
        olddims = vals[:axis] + [0] + vals[axis:]
        ret = np.transpose(ret,tuple(olddims))
        return ret
    else:
        raise ValueError('Bad shapes: x: %s, y: %s' % (x.shape, y.shape))


def DerNoise1D(flux, masked=False,out_signal=False):
    '''
    DESCRIPTION
    
    This function computes the signal to noise ratio DER_SNR following the
    definition set forth by the Spectral Container Working Group of ST-ECF,
    MAST and CADC. 

        noise  = 1.482602 / sqrt(6) median(abs(2 flux_i - flux_i-2 - flux_i+2))
        values with padded zeros are skipped

    NOTES
    
    The DER_SNR algorithm is an unbiased estimator describing the spectrum 
    as a whole as long as
        * the noise is uncorrelated in wavelength bins spaced two pixels apart
        * the noise is Normal distributed
        * for large wavelength regions, the signal over the scale of 5 or
        more pixels can be approximated by a straight line

    For most spectra, these conditions are met.

    REFERENCES

    * ST-ECF Newsletter, Issue #42:
        www.spacetelescope.org/about/further_information/newsletters/html/newsletter_42.html
    * Software:
        www.stecf.org/software/ASTROsoft/DER_SNR/

    AUTHOR
    
    Felix Stoehr, ST-ECF | Adapted to masked arrays by RGB
    '''
    if type(flux) is np.ma.core.MaskedArray and not masked:
        data = flux.data.copy()
    else:
        data = flux.copy()
    data = np.array(data[np.where(data != 0.0)])
    n    = len(data)
    
    if n>4:
        noise = (1.482602 / np.sqrt(6)) * np.median(np.abs(2.0 * data[2:n-2] - data[0:n-4] - data[4:n]))
    else:
        noise = 0.0
    if out_signal:
        if n < 1: signal = 0.0
        else:     signal = np.median(data)
        return signal, noise
    else:
        return noise
