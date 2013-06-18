'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
import numpy as np
import matplotlib.pyplot as plt
import scipy


def disk(I_D0, R_0, r):
    return I_D0 * np.exp(-r / R_0)

def disk_deriv(I_D0, R_0, r):
    I_D = disk(r, I_D0, R_0)
    d_I_D0 = I_D / I_D0
    d_R0 = I_D / R_0**2 * r
    return d_I_D0, d_R0 

def bulge(I_Be, R_e, r):
    return I_Be * np.exp(-7.669 * ((r / R_e)**0.25 - 1))

def bulge_deriv(I_Be, R_e, r):
    I_B = bulge(r, I_Be, R_e)
    d_I_Be = I_B / I_Be
    d_Re = I_B * (7.669 / 4.0 / R_e**1.25) * r**0.25
    return d_I_Be, d_Re


def galaxy(params, r):
    I_Be, R_e, I_D0, R_0 = params
    _I_B = bulge(I_Be, R_e, r)
    _I_D = disk(I_D0, R_0, r)
    return _I_B + _I_D

def galaxy_deriv(params, r, y):
    I_Be, R_e, I_D0, R_0 = params
    d_I_Be, d_Re = bulge_deriv(I_Be, R_e, r)
    d_I_D0, d_R0 = disk_deriv(I_D0, R_0, r)
    return np.array([d_I_Be, d_Re, d_I_D0, d_R0]).T 

def residual(params, r, data):
    if (params <= 0.0).any():
        return np.ones_like(r)
    return data - galaxy(params, r)

def fit(data, r, I_Be, R_e, I_D0, R_0):
    return scipy.optimize.leastsq(residual, x0=np.array((I_Be, R_e, I_D0, R_0)), args=(r, data))

K = fitsQ3DataCube('../../../cubes.200/K0127_synthesis_eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits')
pa, ba = K.getEllipseParams()
K.setGeometry(pa, ba)

Nl = 4
Nr = 40

dl = int(np.floor(K.Nl_obs / Nl))
bins_l = K.l_obs[np.arange(0, K.Nl_obs, dl)]
l = K.l_obs[np.arange(dl/2, K.Nl_obs-dl/2, dl)]

pa__l = np.empty(Nl)
ba__l = np.empty(Nl)
radprof__lr = np.empty((Nl, Nr))
plt.figure(1)
plt.clf()
gs = plt.GridSpec(2,2)
bins_r = np.arange(1, Nr+1)


for i in xrange(Nl):
    print i
    l1 = i*dl
    l2 = (i+1)*dl
    fl = K.f_syn__lyx[l1:l2].sum(axis=0) / dl
    pa__l[i], ba__l[i] = K.getEllipseParams(fl)
    K.setGeometry(pa__l[i], ba__l[i])
    mask = K.qMask & (K.pixelDistance__yx >= 1.0)
    rl = K.pixelDistance__yx[mask]
    sort_ix = np.argsort(rl)
    rl = rl[sort_ix]
    fl = fl[mask][sort_ix] / 1e-17

    print    
    print    
    print    
    print 'Guess: '
    print 'I_Be=',fl.max(),', R_e=',K.HLR_pix,', I_D0=',fl.max()/2.0,', R_0=',K.HLR_pix
    fit_params, _ = fit(fl, rl, I_Be=fl.max(), R_e=K.HLR_pix, I_D0=fl.max()/2.0, R_0=K.HLR_pix)
    I_Be, R_e, I_D0, R_0 = fit_params
    print 'Fit:'
    print 'I_Be=',I_Be, ', R_e=',R_e, ', I_D0=', I_D0, ', R_0=', R_0
    ax = plt.subplot(gs[i])
    ax.plot(rl, np.log10(fl), 'b.')
    ax.plot(bins_r, np.log10(galaxy(fit_params, bins_r)), 'r-')
    ax.plot(bins_r, np.log10(bulge(I_Be, R_e, bins_r)), 'r:')
    ax.plot(bins_r, np.log10(disk(I_D0, R_0, bins_r)), 'r--')
#     ax.set_ylim(-1, 2.5)
    ax.set_title(r'$\lambda = %d \AA$' % l[i])
    ax.set_xlabel(r'$r\ [arcsec]$')
    ax.set_ylabel(r'$\log F_\lambda [erg / s / cm^2 / \AA]$')
