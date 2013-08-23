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


def get_sperf_sum(sper_arr, spef_arr, nz, beta, fg=2.0):
    '''
    Get spectrum error of the sum of zones.
    '''
    # Creating flag array
    tgflag = np.where(spef_arr > 0, 0.0 , 1.0)  # Flag good pixels (1)
    tzflag = np.where(spef_arr > 0, 1.0, 0.0)  # Flag bad pixels  (1)
    
    # Number of good spectra per lambda
    tsgflag = tgflag.sum(axis=1)
    # If all spectra are bad at that lambda (0), replace by total number of spectra (ns)
    _id = tsgflag < 1.0
    tsgflag[_id] = nz
    tgflag[_id] = 1.0
    # tszflag = number of bad spectra per lambda --> Apply rule "botella medio llena"
    tszflag = tzflag.sum(axis=1)
    tszflag = np.where(tszflag >= ((nz / fg) + 0.5), 1.0, 0.0)
    
    # Sum of errors and flags
    sper_sum = np.sqrt((np.power(sper_arr, 2.0) * tgflag * beta * beta).sum(axis=1) * nz / tsgflag)
    spef_sum = spef_arr.sum(axis=1) * tszflag  # Total flag * Rule "botella medio llena"
    return sper_sum, spef_sum


def integrated_spec(f_obs, f_err, f_flag):
    N_zone = f_obs.shape[1]
    beta__z = areafactor(N_zone)
    # FIXME: how to sum f_obs?
    i_f_obs = f_obs.sum(axis=1)
    i_f_err, i_f_flag = get_sperf_sum(f_err, f_flag, N_zone, beta__z)
    # Flag assignment
    i_f_flag[np.where(i_f_flag > 0)] = inf
    fber_sum = np.where(i_f_err > tbfe * abs(i_f_obs), ibe, 0.0)
    sm_err = np.median(i_f_err[np.where((i_f_obs > 0.0) & (i_f_err > 0.0) & (fber_sum < 1.0) & (i_f_flag < 1.0))])
    fser_sum = np.where(i_f_err < tsfe * sm_err, ise, 0.0)  # Flag for small Errors
    i_f_flag = i_f_flag + fber_sum + fser_sum  # Total flag
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
