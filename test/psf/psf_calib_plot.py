'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
import matplotlib.pyplot as plt
import glob
from os import path
import sys

func = sys.argv[2] # 'Kolmogorov', 'Moffat, 'Gaussian'
beta4 = len(sys.argv) > 3 and sys.argv[3] == 'beta4'
if beta4:
    name = '%sBeta4' % (func)
else:
    name = func

galaxiesV500 = glob.glob(path.join(sys.argv[1], '%s_[a-zA-Z0-9]*.[0-9]*.V500.v1.5.PSF.dat' % name))
print galaxiesV500
param_dtype = [('lambda', 'float64'), ('I_0', 'float64'),
               ('fwhm', 'float64'), ('beta', 'float64'), 
               ('x0', 'float64'), ('y0', 'float64'),
               ('PA', 'float64'), ('ell', 'float64'),
               ('good', 'float64'), ('flag', 'bool'), ('chi2', 'float64')]

nlambdaV500 = 10
nlambdaV1200 = 30

fwhmV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
betaV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
x0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
y0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
ellV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
chi2V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))

#===============================================================================
# fwhmV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
# betaV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
# x0V1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
# y0V1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
# ellV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
#===============================================================================

for i, galaxy in enumerate(galaxiesV500):
    p = np.genfromtxt(galaxy, dtype=param_dtype)
    wlV500 = p['lambda']
    mask = p['flag'] | (p['good'] < 0.6)
    fwhmV500[i] = p['fwhm']
    fwhmV500[i, mask] = np.ma.masked
    betaV500[i] = p['beta']
    betaV500[i, mask] = np.ma.masked
    x0V500[i] = p['x0']
    x0V500[i, mask] = np.ma.masked
    y0V500[i] = p['y0']
    y0V500[i, mask] = np.ma.masked
    ellV500[i] = p['ell']
    ellV500[i, mask] = np.ma.masked
    chi2V500[i] = p['chi2']
    chi2V500[i, mask] = np.ma.masked

#===============================================================================
# for i, galaxy in enumerate(galaxiesV1200):
#     cube = 'psf/%s.%s.v1.5.PSF.dat' % (galaxy, 'V1200')
#     p = np.genfromtxt(cube, dtype=param_dtype)
#     wlV1200 = p['lambda']
#     mask = p['flag'] | (p['good'] < 0.7)
#     fwhmV1200[i] = p['fwhm']
#     fwhmV1200[i, mask] = np.ma.masked
#     betaV1200[i] = p['beta']
#     betaV1200[i, mask] = np.ma.masked
#     x0V1200[i] = p['x0']
#     x0V1200[i, mask] = np.ma.masked
#     y0V1200[i] = p['y0']
#     y0V1200[i, mask] = np.ma.masked
#     ellV1200[i] = p['ell']
#     ellV1200[i, mask] = np.ma.masked
#===============================================================================


fwhmV500_b = (fwhmV500 * (1.0 - ellV500))
wlmin = wlV500.min()
wlmax = wlV500.max()

def getstats1(p, wei):
    p_wei = np.sum(p * wei)
    p_var = np.sum((p - p_wei)**2 * wei)
    p_std = np.sqrt(p_var)
    return p_wei, p_std

def getstats(p, wei):
    p_wei = np.sum(p * wei, axis=0)
    p_var = np.sum((p - p_wei)**2 * wei, axis=0)
    p_std = np.sqrt(p_var)
    return p_wei, p_std

wei = np.exp(-0.5 * chi2V500 / chi2V500.min())
wei /= np.sum(wei, axis=0)
wei1 = wei / wei.sum()

fwhmV500_wei, fwhmV500_std = getstats(fwhmV500, wei)
fwhmbV500_wei, fwhmbV500_std = getstats(fwhmV500_b, wei)
betaV500_wei, betaV500_std = getstats(betaV500, wei)

plt.clf()
plt.subplot(211)
plt.plot(wlV500, fwhmV500_wei, 'r-')
plt.plot(wlV500, fwhmV500_wei - fwhmV500_std, 'r--')
plt.plot(wlV500, fwhmV500_wei + fwhmV500_std, 'r--')
plt.plot(wlV500, fwhmbV500_wei, ls='-', color='pink')
plt.plot(wlV500, fwhmbV500_wei - fwhmbV500_std, ls='--', color='pink')
plt.plot(wlV500, fwhmbV500_wei + fwhmbV500_std, ls='--', color='pink')
plt.ylabel(r'FWHM $[arcsec]$')
if func != 'Moffat' or beta4:
    plt.xlabel(r'wavelength $[\AA]$')
else:
    plt.gca().set_xticklabels([])
plt.ylim(0.0, 5.0)
plt.xlim(wlmin, wlmax)

if func == 'Moffat' and not beta4:
    plt.subplot(212)
    plt.plot(wlV500, betaV500_wei, 'r-')
    plt.plot(wlV500, betaV500_wei - betaV500_std, 'r--')
    plt.plot(wlV500, betaV500_wei + betaV500_std, 'r--')
    plt.ylabel(r'$\beta$')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylim(0.0, 4.0)
    plt.xlim(wlmin, wlmax)
    
if func == 'Moffat' and beta4:
    plt.suptitle(r'%s | $\beta=4$ | $\mathrm{FWHM}=%.3f \pm %.3f$' % ((func,) + getstats1(fwhmV500, wei1)))
elif func == 'Moffat':
    plt.suptitle(r'%s | $\beta=%.3f \pm %.3f$ | $\mathrm{FWHM}=%.3f \pm %.3f$' % ((func,) + getstats1(betaV500, wei1) + getstats1(fwhmV500, wei1)))
else:
    plt.suptitle(r'%s | $\mathrm{FWHM}=%.3f \pm %.3f$' % ((func,) + getstats1(fwhmV500, wei1)))
plt.savefig(path.join(sys.argv[1], '%s_PSF_all.png' % name))

print 'Summary:'
print 'fwhm(a) = %.3f +- %.3f' % getstats1(fwhmV500, wei1)
print 'fwhm(b) = %.3f +- %.3f' % getstats1(fwhmV500_b, wei1)
if func == 'Moffat' and not beta4:
    print 'beta = %.3f +- %.3f' % getstats1(betaV500, wei1)
print 'ell = %.3f +- %.3f' % getstats1(ellV500, wei1)

