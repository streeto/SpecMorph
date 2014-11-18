'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
from os import path

func = sys.argv[2] # 'Kolmogorov', 'Moffat, 'Gaussian'
beta4 = len(sys.argv) > 3 and sys.argv[3] == 'beta4'
if beta4:
    name = '%sBeta4' % (func)
else:
    name = func

galaxiesV500 = ['NGC0023',
                'NGC2916',
                'IC1652',
                'NGC0447',
                'NGC5473',
                'NGC5557',
                'NGC5602',
                'NGC5642',
                'NGC6762']

galaxiesV1200 = ['NGC0023',
                 'NGC2916',
                 'IC1652',
                 'NGC0447',
                 'NGC6762']

psfcenter = {'NGC0023': (14, 16),
             'NGC2916': (45, 32),
             #'NGC0180': (62, 57),
             'IC1652': (52, 36),
             'NGC0447': (22, 16),
             'NGC5473': (45, 14),
             'NGC5557': (18, 24),
             'NGC5602': (40, 23),
             'NGC5642': (25, 18),
             'NGC6762': (22, 20),
             }

param_dtype = [('lambda', 'float64'), ('I_0', 'float64'),
               ('fwhm', 'float64'), ('beta', 'float64'), 
               ('x0', 'float64'), ('y0', 'float64'),
               ('PA', 'float64'), ('ell', 'float64'),
               ('good', 'float64'), ('flag', 'float64'), ('chi2', 'float64')]

nlambdaV500 = 10
nlambdaV1200 = 3

fwhmV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
betaV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
x0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
y0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
ellV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
chi2V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))

fwhmV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
betaV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
x0V1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
y0V1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
ellV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
chi2V1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))

for i, galaxy in enumerate(galaxiesV500):
    cube = path.join(sys.argv[1], '%s_%s.%s.v1.5.PSF.dat' % (name, galaxy, 'V500'))
    p = np.genfromtxt(cube, dtype=param_dtype)
    wlV500 = p['lambda']
    mask = p['flag'] > 0.0
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

for i, galaxy in enumerate(galaxiesV1200):
    cube = 'out/%s_%s.%s.v1.5.PSF.dat' % (name, galaxy, 'V1200')
    p = np.genfromtxt(cube, dtype=param_dtype)
    wlV1200 = p['lambda']
    mask = p['flag'] > 0.0
    fwhmV1200[i] = p['fwhm']
    fwhmV1200[i, mask] = np.ma.masked
    betaV1200[i] = p['beta']
    betaV1200[i, mask] = np.ma.masked
    x0V1200[i] = p['x0']
    x0V1200[i, mask] = np.ma.masked
    y0V1200[i] = p['y0']
    y0V1200[i, mask] = np.ma.masked
    ellV1200[i] = p['ell']
    ellV1200[i, mask] = np.ma.masked
    chi2V1200[i] = p['chi2']
    chi2V1200[i, mask] = np.ma.masked


wlmin = wlV1200.min()
wlmax = wlV500.max()
imcenter = (36, 38.5)

distV500 = []
for g in galaxiesV500:
    x = psfcenter[g]
    distV500.append(np.sqrt((x[0] - imcenter[0])**2 + (x[1] - imcenter[1])**2))
distV1200 = []
for g in galaxiesV1200:
    x = psfcenter[g]
    distV1200.append(np.sqrt((x[0] - imcenter[0])**2 + (x[1] - imcenter[1])**2))

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

weiV500 = np.exp(-0.25 * chi2V500 / chi2V500.min())
weiV500 /= np.sum(weiV500, axis=0)
weiV500_1 = weiV500 / weiV500.sum()
weiV1200 = np.exp(-0.25 * chi2V1200 / chi2V1200.min())
weiV1200 /= np.sum(weiV1200, axis=0)
weiV1200_1 = weiV1200 / weiV1200.sum()

fwhmV500_wei, fwhmV500_std = getstats(fwhmV500, weiV500)
betaV500_wei, betaV500_std = getstats(betaV500, weiV500)
fwhmV1200_wei, fwhmV1200_std = getstats(fwhmV1200, weiV1200)
betaV1200_wei, betaV1200_std = getstats(betaV1200, weiV1200)

plt.clf()
plt.subplot(211)
plt.plot(wlV500, fwhmV500_wei, 'r-')
plt.plot(wlV500, fwhmV500_wei - fwhmV500_std, 'r--')
plt.plot(wlV500, fwhmV500_wei + fwhmV500_std, 'r--')
plt.plot(wlV1200, fwhmV1200_wei, 'b-')
plt.plot(wlV1200, fwhmV1200_wei + fwhmV1200_std, 'b--')
plt.plot(wlV1200, fwhmV1200_wei - fwhmV1200_std, 'b--')
plt.text(6000, 3.5, r'$\mathrm{FWHM(V500)}=%.3f \pm %.3f$' % getstats1(fwhmV500, weiV500_1))
plt.text(4000, 3.5, r'$\mathrm{FWHM(V1200)}=%.3f \pm %.3f$' % getstats1(fwhmV1200, weiV1200_1))
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
    plt.plot(wlV1200, betaV1200_wei, 'b-')
    plt.plot(wlV1200, betaV1200_wei + betaV1200_std, 'b--')
    plt.plot(wlV1200, betaV1200_wei - betaV1200_std, 'b--')
    plt.text(6000, 3.5, r'$\beta\mathrm{(V500)}=%.3f \pm %.3f$' % getstats1(betaV500, weiV500_1))
    plt.text(4000, 3.5, r'$\beta\mathrm{(V500)}=%.3f \pm %.3f$' % getstats1(betaV500, weiV500_1))
    plt.ylabel(r'$\beta$')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylim(0.0, 4.0)
    plt.xlim(wlmin, wlmax)

if func == 'Moffat' and beta4:
    plt.suptitle(r'%s | $\beta=4$' % func)
else:
    plt.suptitle(r'%s' % func)

plt.savefig(path.join(sys.argv[1], '%s_PSF_all.png' % name))


plt.figure(figsize=(5,6))
plt.clf()

plt.subplot(211)
plt.plot(distV500, fwhmV500.mean(axis=1), 'ro', label='V500')
plt.plot(distV1200, fwhmV1200.mean(axis=1), 'bo', label='V1200')
plt.ylabel(r'FWHM $[arcsec]$')
#plt.xlabel(r'distance $[arcsec]$')
plt.gca().set_xticklabels([])
plt.ylim(0.0,5.0)
plt.xlim(0.0,35.0)
plt.legend(loc='lower right')

if func == 'Moffat' and not beta4:
    plt.subplot(212)
    plt.plot(distV500, betaV500.mean(axis=1), 'ro', label='V500')
    plt.plot(distV1200, betaV1200.mean(axis=1), 'bo', label='V1200')
    plt.ylabel(r'$\beta$')
    plt.xlabel(r'distance $[arcsec]$')
    plt.ylim(0.0,5.0)
    plt.xlim(0.0,35.0)

plt.savefig(path.join(sys.argv[1], '%s_PSF_distance.png' % name))

print 'Summary:'
print 'fwhm(V500) = %.3f +- %.3f' % getstats1(fwhmV500, weiV500_1)
print 'fwhm(V1200) = %.3f +- %.3f' % getstats1(fwhmV1200, weiV1200_1)
if func == 'Moffat' and not beta4:
    print 'beta(V500) = %.3f +- %.3f' % getstats1(betaV500, weiV500_1)
    print 'beta(V1200) = %.3f +- %.3f' % getstats1(betaV1200, weiV1200_1)
print 'ell(V500) = %.3f +- %.3f' % getstats1(ellV500, weiV500_1)
print 'ell(V1200) = %.3f +- %.3f' % getstats1(ellV1200, weiV1200_1)

