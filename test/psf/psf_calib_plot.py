'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
import matplotlib.pyplot as plt
import glob
from os import path
import sys

func = sys.argv[3] # 'Kolmogorov', 'Moffat, 'Gaussian'
beta4 = len(sys.argv) > 3 and sys.argv[3] == 'beta4'
if beta4:
    name = '%s_beta4' % (func)
else:
    name = func

galaxiesV500 = glob.glob(path.join(sys.argv[1], '%s_*.V500.v1.5.PSF.dat' % name))
print galaxiesV500
param_dtype = [('lambda', 'float64'), ('I_0', 'float64'),
               ('fwhm', 'float64'), ('beta', 'float64'), 
               ('x0', 'float64'), ('y0', 'float64'),
               ('PA', 'float64'), ('ell', 'float64'),
               ('good', 'float64'), ('flag', 'bool'), ('chi2', 'float64')]

nlambdaV500 = 94
nlambdaV1200 = 30

fwhmV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
betaV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
x0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
y0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
ellV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))

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

plt.clf()
plt.subplot(211)
plt.plot(wlV500, fwhmV500.mean(axis=0), 'r-')
plt.plot(wlV500, fwhmV500.mean(axis=0) - fwhmV500.std(axis=0), 'r--')
plt.plot(wlV500, fwhmV500.mean(axis=0) + fwhmV500.std(axis=0), 'r--')
plt.plot(wlV500, fwhmV500_b.mean(axis=0), ls='-', color='pink')
plt.plot(wlV500, fwhmV500_b.mean(axis=0) - fwhmV500_b.std(axis=0), ls='--', color='pink')
plt.plot(wlV500, fwhmV500_b.mean(axis=0) + fwhmV500_b.std(axis=0), ls='--', color='pink')
#plt.plot(wlV1200, fwhmV1200.mean(axis=0), 'b-')
#plt.plot(wlV1200, fwhmV1200.mean(axis=0) + fwhmV1200.std(axis=0), 'b--')
#plt.plot(wlV1200, fwhmV1200.mean(axis=0) - fwhmV1200.std(axis=0), 'b--')
plt.ylabel(r'FWHM $[arcsec]$')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylim(0.0, 5.0)
plt.xlim(wlmin, wlmax)

if func == 'Moffat' and not beta4:
    plt.subplot(212)
    plt.plot(wlV500, betaV500.mean(axis=0), 'r-')
    plt.plot(wlV500, betaV500.mean(axis=0) - betaV500.std(axis=0), 'r--')
    plt.plot(wlV500, betaV500.mean(axis=0) + betaV500.std(axis=0), 'r--')
    #plt.plot(wlV1200, betaV1200.mean(axis=0), 'b-')
    #plt.plot(wlV1200, betaV1200.mean(axis=0) + betaV1200.std(axis=0), 'b--')
    #plt.plot(wlV1200, betaV1200.mean(axis=0) - betaV1200.std(axis=0), 'b--')
    plt.ylabel(r'$\beta$')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylim(0.0, 4.0)
    plt.xlim(wlmin, wlmax)

plt.savefig(path.join(sys.argv[1], '%s_PSF_all.png' % name))

print 'Summary:'
print 'fwhm(a) = %.3f +- %.3f' % (fwhmV500.mean(), fwhmV500.std())
print 'fwhm(b) = %.3f +- %.3f' % (fwhmV500_b.mean(), fwhmV500_b.std())
if func == 'Moffat' and not beta4:
    print 'beta = %.3f +- %.3f' % (betaV500.mean(), betaV500.std())
print 'ell = %.3f +- %.3f' % (ellV500.mean(), ellV500.std())

