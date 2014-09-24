'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
import matplotlib.pyplot as plt

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
               ('good', 'float64'), ('flag', 'bool'), ('chi2', 'float64')]

nlambdaV500 = 94
nlambdaV1200 = 30

fwhmV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
betaV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
x0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
y0V500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))
ellV500 = np.ma.empty((len(galaxiesV500),  nlambdaV500))

fwhmV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
betaV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
x0V1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
y0V1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))
ellV1200 = np.ma.empty((len(galaxiesV1200),  nlambdaV1200))

for i, galaxy in enumerate(galaxiesV500):
    cube = 'out/%s.%s.v1.5.PSF.dat' % (galaxy, 'V500')
    p = np.genfromtxt(cube, dtype=param_dtype)
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

for i, galaxy in enumerate(galaxiesV1200):
    cube = 'out/%s.%s.v1.5.PSF.dat' % (galaxy, 'V1200')
    p = np.genfromtxt(cube, dtype=param_dtype)
    wlV1200 = p['lambda']
    mask = p['flag'] | (p['good'] < 0.7)
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

plt.clf()
plt.subplot(211)
plt.plot(wlV500, fwhmV500.mean(axis=0), 'r-')
plt.plot(wlV500, fwhmV500.mean(axis=0) - fwhmV500.std(axis=0), 'r--')
plt.plot(wlV500, fwhmV500.mean(axis=0) + fwhmV500.std(axis=0), 'r--')
plt.plot(wlV1200, fwhmV1200.mean(axis=0), 'b-')
plt.plot(wlV1200, fwhmV1200.mean(axis=0) + fwhmV1200.std(axis=0), 'b--')
plt.plot(wlV1200, fwhmV1200.mean(axis=0) - fwhmV1200.std(axis=0), 'b--')
plt.ylabel(r'FWHM $[arcsec]$')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylim(0.0, 5.0)
plt.xlim(wlmin, wlmax)

plt.subplot(212)
plt.plot(wlV500, betaV500.mean(axis=0), 'r-')
plt.plot(wlV500, betaV500.mean(axis=0) - betaV500.std(axis=0), 'r--')
plt.plot(wlV500, betaV500.mean(axis=0) + betaV500.std(axis=0), 'r--')
plt.plot(wlV1200, betaV1200.mean(axis=0), 'b-')
plt.plot(wlV1200, betaV1200.mean(axis=0) + betaV1200.std(axis=0), 'b--')
plt.plot(wlV1200, betaV1200.mean(axis=0) - betaV1200.std(axis=0), 'b--')
plt.ylabel(r'$\beta$')
plt.xlabel(r'wavelength $[\AA]$')
plt.ylim(0.0, 4.0)
plt.xlim(wlmin, wlmax)

plt.savefig('out/PSF_all.png')


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

plt.subplot(212)
plt.plot(distV500, betaV500.mean(axis=1), 'ro', label='V500')
plt.plot(distV1200, betaV1200.mean(axis=1), 'bo', label='V1200')
plt.ylabel(r'$\beta$')
plt.xlabel(r'distance $[arcsec]$')
plt.ylim(0.0,5.0)
plt.xlim(0.0,35.0)
plt.savefig('out/PSF_distance.png')
