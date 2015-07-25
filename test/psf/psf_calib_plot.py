# -*- coding: utf-8 -*-
'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
from pylab import normpdf
import matplotlib.pyplot as plt
import glob
from os import path
import sys

################################################################################
def plot_setup():
    plotpars = {'legend.fontsize': 8,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.fontsize': 10,
                'axes.titlesize': 12,
                'lines.linewidth': 0.5,
                'font.family': 'Times New Roman',
    #             'figure.subplot.left': 0.08,
    #             'figure.subplot.bottom': 0.08,
    #             'figure.subplot.right': 0.97,
    #             'figure.subplot.top': 0.95,
    #             'figure.subplot.wspace': 0.42,
    #             'figure.subplot.hspace': 0.1,
                'image.cmap': 'GnBu',
                }
    plt.rcParams.update(plotpars)
    plt.ioff()
################################################################################

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

fwhmV500_wei1, fwhmV500_std1 = getstats1(fwhmV500, wei1)
fwhmbV500_wei1, fwhmbV500_std1 = getstats1(fwhmV500_b, wei1)
betaV500_wei1, betaV500_std1 = getstats1(betaV500, wei1)

plot_setup()
width_pt = 448.07378
width_in = width_pt / 72.0 * 0.9
fig = plt.figure(figsize=(width_in, width_in * 1.0))
gs = plt.GridSpec(2, 1, height_ratios=[1.0, 1.0])
ax = plt.subplot(gs[0])
ax.plot(wlV500, fwhmV500_wei, 'ko-', mfc='none')
ax.plot(wlV500, fwhmV500_wei - fwhmV500_std, 'k--')
ax.plot(wlV500, fwhmV500_wei + fwhmV500_std, 'k--')
#ax.plot(wlV500, fwhmbV500_wei, ls='-', color='pink')
#ax.plot(wlV500, fwhmbV500_wei - fwhmbV500_std, ls='--', color='pink')
#ax.plot(wlV500, fwhmbV500_wei + fwhmbV500_std, ls='--', color='pink')
ax.set_ylabel(r'FWHM $[\mathrm{arcsec}]$')
if func != 'Moffat' or beta4:
    ax.set_xlabel(r'Comprimento de onda $[\AA]$')
else:
    ax.set_xticklabels([])
ax.set_ylim(0.0, 4.0)
#ax.set_xlim(wlmin, wlmax)
ax.set_xlim(3700, 7500)

if func == 'Moffat' and not beta4:
    plt.subplot(212)
    plt.plot(wlV500, betaV500_wei, 'r-')
    plt.plot(wlV500, betaV500_wei - betaV500_std, 'r--')
    plt.plot(wlV500, betaV500_wei + betaV500_std, 'r--')
    plt.ylabel(r'$\beta$')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylim(0.0, 4.0)
    #plt.xlim(wlmin, wlmax)
    plt.xlim(3700, 7500)
else:
    nsigma = 2.5
    nbin = 10
    ax = plt.subplot(gs[1])
    r = [fwhmV500_wei1 - nsigma * fwhmV500_std1, fwhmV500_wei1 + nsigma * fwhmV500_std1]
    ax.hist(fwhmV500.compressed(), weights=wei.compressed(), bins=nbin, range=r, normed=True,
            color='k', histtype='step')
    x = np.linspace(r[0], r[1], 100)
    ax.plot(x, normpdf(x, fwhmV500_wei1, fwhmV500_std1), 'k--')
    ax.vlines(fwhmV500_wei1, ymin=0, ymax=2, color='k', linestyles='--')
    ax.text(fwhmV500_wei1 - 0.05, 1.4, r'$\mathrm{FWHM}\ =\ %.3f\,^{\prime\prime}\pm\, %.3f$' % (fwhmV500_wei1, fwhmV500_std1), ha='right')
    #ax.set_xlim(r[0], r[1])
    ax.set_xlim(1.5, 4)
    ax.set_ylim(0, 1.6)
    ax.set_ylabel(r'Densidade de probabilidade')
    ax.set_xlabel(r'FWHM $[\mathrm{arcsec}]$')

if func == 'Moffat' and beta4:
    plt.suptitle(u'Perfil de Moffat - estrelas de calibração')
elif func == 'Moffat':
    plt.suptitle(r'%s | $\beta=%.3f \pm %.3f$ | $\mathrm{FWHM}=%.3f \pm %.3f$' % ((func,) + getstats1(betaV500, wei1) + getstats1(fwhmV500, wei1)))
else:
    plt.suptitle(r'%s | $\mathrm{FWHM}=%.3f \pm %.3f$' % ((func,) + getstats1(fwhmV500, wei1)))

gs.tight_layout(fig, rect=[0, 0, 1, 0.97])    
plt.savefig(path.join(sys.argv[1], '%s_PSF_all.pdf' % name))

print 'Summary:'
print 'fwhm(a) = %.3f +- %.3f' % (fwhmV500_wei1, fwhmV500_std1)
print 'fwhm(b) = %.3f +- %.3f' % (fwhmbV500_wei1, fwhmbV500_std1)
if func == 'Moffat' and not beta4:
    print 'beta = %.3f +- %.3f' % (betaV500_wei1, betaV500_std1)
print 'ell = %.3f +- %.3f' % getstats1(ellV500, wei1)

