'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pylab import normpdf
import sys


def getstats(p, wei):
    p_wei = np.sum(p * wei)
    p_var = np.sum((p - p_wei)**2 * wei)
    p_std = np.sqrt(p_var)
    return p_wei, p_std

func = sys.argv[1]
beta4 = len(sys.argv) > 2 and sys.argv[2] == 'beta4'
if beta4:
    name = '%s_beta4' % (func)
else:
    name = func

param_dtype = [('star', 'S40'), ('I_0', 'float64'), 
               ('fwhm', 'float64'), ('beta', 'float64'), 
               ('x0', 'float64'), ('y0', 'float64'), 
               ('PA', 'float64'), ('ell', 'float64'), 
               ('good', 'float64'), ('flag', 'int'), ('chi2', 'float64')]
params = np.genfromtxt('test/psf/out_calib_new/%s_fit_gband.dat' % name, dtype=param_dtype)
params = np.ma.array(params, mask=params['flag'] > 0)

wei = np.exp(-0.5 * params['chi2'])
wei /= np.sum(wei)
fwhm_wei, fwhm_std = getstats(params['fwhm'], wei)
beta_wei, beta_std = getstats(params['beta'], wei)
ell_wei, ell_std = getstats(params['ell'], wei)

print 'FWHM = %.3f +- %.3f' % (fwhm_wei, fwhm_std)
if func == 'Moffat':
    print 'beta = %.3f +- %.3f' % (beta_wei, beta_std)
print 'ell = %.3f +- %.3f' % (ell_wei, ell_std)

nsigma = 2.5
nbin = 10
plt.figure(1, figsize=(4, 5))
plt.clf()
plt.subplot(211)
r = [fwhm_wei - nsigma * fwhm_std, fwhm_wei + nsigma * fwhm_std]
plt.hist(params['fwhm'].compressed(), weights=wei.compressed(), bins=nbin, range=r, normed=True, histtype='step')
x = np.linspace(r[0], r[1], 100)
plt.plot(x, normpdf(x, fwhm_wei, fwhm_std))
plt.vlines(fwhm_wei, ymin=0, ymax=2, linestyles='--')
plt.text(0.37, 0.1, '%.3f' % fwhm_wei, transform=plt.gca().transAxes)
plt.xlim(r[0], r[1])
plt.ylim(0, 2)
plt.xlabel(r'FWHM $[arcsec]$')

if func == 'Moffat' and not beta4:
    plt.subplot(212)
    r = [beta_wei - nsigma * beta_std, beta_wei + nsigma * beta_std]
    plt.hist(params['beta'].compressed(), weights=wei.compressed(), bins=nbin, range=r, normed=True, histtype='step')
    x = np.linspace(r[0], r[1], 100)
    plt.plot(x, normpdf(x, beta_wei, beta_std))
    plt.vlines(beta_wei, ymin=0, ymax=6, linestyles='--')
    plt.text(0.37, 0.1, '%.3f' % beta_wei, transform=plt.gca().transAxes)
    plt.xlabel(r'$\beta$')
    plt.xlim(r[0], r[1])
    plt.ylim(0, 6)

plt.suptitle(func)
plt.tight_layout()
plt.savefig('test/psf/out_calib_new/%s_PSF_gband.png' % name)
