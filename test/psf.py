'''
Created on 10/09/2014

@author: andre
'''

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from pycasso.util import radialProfile, fillImage, getEllipseParams
from specmorph.geometry import fix_PA_ell
from imfit import Imfit, SimpleModelDescription, function_description
import sys

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

galaxycenter = {'NGC0023': (31.4, 35.6),
                'NGC2916': (32.6, 35.3),
                #'NGC0180': (31.8, 33.1),
                'IC1652': (20.2, 33.2),
                'NGC0447': (34.0, 33.5),
                'NGC5473': (32.2, 33.1),
                'NGC5557': (33.2, 32.1),
                'NGC5602': (31.3, 33.3),
                'NGC5642': (31.2, 33.7),
                'NGC6762': (32.2, 38.2),
                }

galaxy = sys.argv[1]
cubeversion = sys.argv[2]
cube = '../../cubes.DR2/%s.%s.rscube.fits.gz' % (galaxy, cubeversion)
psfradius = 7
psfLstep = 40.0 # /AA
badpix_frac = 0.5

debug = False
i_plot = 3

f = pyfits.open(cube)
lambda0 = f[0].header['CRVAL3']
dlambda = f[0].header['CDELT3']
_mask = f[3].data
_flux = np.ma.array(f[0].data, mask=_mask)
wl = np.arange(lambda0, lambda0 + (_flux.shape[0]) * dlambda, dlambda)
f.close()

psfLstep_ix = int(np.ceil(psfLstep / dlambda))
wl_ix = np.arange(0, len(wl), psfLstep_ix)
flux = np.ma.empty((len(wl_ix), _flux.shape[1], _flux.shape[2]))
mask = np.ma.empty((len(wl_ix), _flux.shape[1], _flux.shape[2]), dtype='bool')
for i, j in enumerate(wl_ix):
    print 'Computing binned image %d of %d (@ %.1f \\AA)' % (i, len(wl_ix), wl[j])
    l_radius = psfLstep_ix / 2
    j1 = j - l_radius if j > (0 + l_radius) else 0
    j2 = j + l_radius if j < (wl_ix[-1] - l_radius) else wl_ix[-1]
    mask[i] = _mask[j1:j2].sum(axis=0) > badpix_frac * (j2 - j1)
    flux[i] = np.ma.median(_flux[j1:j2], axis=0)

psfYrange = slice(psfcenter[galaxy][0] - psfradius, psfcenter[galaxy][0] + psfradius + 1)
psfXrange = slice(psfcenter[galaxy][1] - psfradius, psfcenter[galaxy][1] + psfradius + 1)

psfflux = flux[:,psfYrange, psfXrange].copy()
fill_flux = flux
psf_wl = wl[wl_ix]

if debug:
    plt.clf()
    plt.ioff()
    plt.imshow(fill_flux[i_plot])
    plt.colorbar()
    plt.show()


fill_flux[:, psfYrange, psfXrange] = np.ma.masked

if debug:
    plt.clf()
    plt.ioff()
    plt.imshow(fill_flux[i_plot])
    plt.colorbar()
    plt.show()

for i in xrange(fill_flux.shape[0]):
    print 'Filling %d of %d' % (i, fill_flux.shape[0])
    pa, ba = getEllipseParams(fill_flux[i], galaxycenter[galaxy][1], galaxycenter[galaxy][0])
    fill_flux[i] = fillImage(fill_flux[i], galaxycenter[galaxy][1], galaxycenter[galaxy][0], pa, ba)

psfbg = fill_flux[:, psfYrange, psfXrange]

if debug:
    plt.clf()
    plt.ioff()
    plt.imshow(fill_flux[i_plot])
    plt.colorbar()
    plt.show()
    
    plt.ioff()
    plt.clf()
    plt.imshow(psfflux[i_plot])
    plt.colorbar()
    plt.show()
    
    plt.ioff()
    plt.clf()
    plt.imshow(psfbg[i_plot])
    plt.colorbar()
    plt.show()
    
    plt.ioff()
    plt.clf()
    plt.imshow(psfflux[i_plot] - psfbg[i_plot])
    plt.colorbar()
    plt.show()

    bins = np.arange(0, psfradius)
    bins_c = bins[:-1] + 0.5
    y0 = psfradius
    x0 = psfradius
    pa = 90.0
    ba = 1.0
    flux_r = radialProfile(psfflux[i_plot], bins, x0, y0, pa, ba)
    bg_r = radialProfile(psfbg[i_plot], bins, x0, y0, pa, ba)
    flux_bg_r = radialProfile(psfflux[i_plot] - psfbg[i_plot], bins, x0, y0, pa, ba)
    
    plt.clf()
    plt.plot(bins_c, flux_r)
    plt.plot(bins_c, flux_bg_r)
    plt.plot(bins_c, bg_r)
    plt.show()

moffat = function_description('Moffat')
moffat.PA.setValue(60, vmin=0, vmax=180)
moffat.ell.setValue(0.2, vmin=0.0, vmax=0.4)
moffat.I_0.setValue(1.0, vmin=1e-20, vmax=10.0)
moffat.fwhm.setValue(2.0, vmin=1e-20, vmax=10.0)
moffat.beta.setValue(1.1, vmin=1e-20, vmax=10.0)
model = SimpleModelDescription()
model.x0.setValue(psfradius)
model.x0.setTolerance(0.2)
model.y0.setValue(psfradius)
model.y0.setTolerance(0.2)
model.addFunction(moffat)

psfmodels = []
psfflags = []
goodfraction = []
chi2 = []
for i in xrange(psfflux.shape[0]):
    print 'Fitting PSF %d of %d' % (i, fill_flux.shape[0])
    imfit = Imfit(model, quiet=True)
    imfit.fit(psfflux[i] - psfbg[i], mode='NM', use_model_for_errors=True)
    _goodfraction = float(imfit.nValidPixels) / float(psfflux[i].size)
    goodfraction.append(_goodfraction)
    psfflags.append(not imfit.fitConverged)
    print '    Reduced fit statistic: %f' % imfit.reducedFitStatistic
    chi2.append(imfit.reducedFitStatistic)
    psfmodels.append(imfit.getModelDescription())

params = np.ma.empty(len(psfmodels), dtype=[('lambda', 'float64'), ('I_0', 'float64'),
                                            ('fwhm', 'float64'), ('beta', 'float64'), 
                                            ('x0', 'float64'), ('y0', 'float64'),
                                            ('PA', 'float64'), ('ell', 'float64'),
                                            ('good', 'float64'), ('flag', 'bool'), ('chi2', 'float64')])

params['lambda'] = psf_wl
params['I_0'] = [m.Moffat.I_0.value for m in psfmodels]
params['fwhm'] = [m.Moffat.fwhm.value for m in psfmodels]
params['beta'] = [m.Moffat.beta.value for m in psfmodels]
params['x0'] = [m.x0.value for m in psfmodels]
params['y0'] = [m.y0.value for m in psfmodels]
PA = []
ell = []
for _PA, _ell in zip([m.Moffat.PA.value for m in psfmodels], [m.Moffat.ell.value for m in psfmodels]):
    _PA, _ell = fix_PA_ell(_PA, _ell)
    PA.append(_PA)
    ell.append(_ell)
params['PA'] = PA
params['ell'] = ell
params['good'] = goodfraction
params['flag'] = psfflags
params['chi2'] = chi2
params[psfflags] = np.ma.masked

header =' '.join(params.dtype.names)
np.savetxt('%s.%s.v1.5.PSF.dat' % (galaxy, cubeversion), params, header=header)

plt.clf()
plt.figure(figsize=(8, 12))
plt.subplot(511)
plt.plot(psf_wl, params['fwhm'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'FWHM $[arcsec]$')
plt.ylim(0.0, 5.0)
plt.xlim(psf_wl.min(), psf_wl.max())
fwhm_median = np.median(params['fwhm'])
plt.hlines(fwhm_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, fwhm_median + 0.075, '%.3f' % fwhm_median)

plt.subplot(512)
plt.plot(psf_wl, params['beta'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'$\beta$')
plt.ylim(0.0, 4.0)
plt.xlim(psf_wl.min(), psf_wl.max())
beta_median = np.median(params['beta'])
plt.hlines(beta_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, beta_median + 0.075, '%.3f' % beta_median)

plt.subplot(513)
plt.plot(psf_wl, params['ell'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'$\epsilon$')
plt.ylim(0.0, 1.05)
plt.xlim(psf_wl.min(), psf_wl.max())
ell_median = np.median(params['ell'])
plt.hlines(ell_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, ell_median, '%.3f' % ell_median)

plt.subplot(514)
plt.plot(psf_wl, PA, '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'P.A. $[degrees]$')
plt.ylim(0.0, 180.0)
plt.xlim(psf_wl.min(), psf_wl.max())
PA_median = np.median(params['PA'])
plt.hlines(PA_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, PA_median, '%.1f' % PA_median)

plt.subplot(515)
plt.plot(psf_wl, params['good'] * 100.0, '-k')
plt.xlabel(r'wavelength $[\AA]$')
plt.ylabel(r'Good pixels $[\%]$')
plt.ylim(0.0, 110.0)
plt.xlim(psf_wl.min(), psf_wl.max())

plt.suptitle('PSF Moffat parameters for %s (%s)' % (galaxy, cubeversion))
plt.savefig('%s.%s.v1.5.PSF.png' % (galaxy, cubeversion))

