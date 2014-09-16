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

cubeversion = 'V500'
galaxy = 'NGC2916'
cube = '../data/%s.%s.rscube.fits.gz' % (galaxy, cubeversion)
galaxycenter = (32.6, 35.3)
psfcenter = (45, 32)
psfradius = 7
psfLstep = 40.0 # /AA
badpix_frac = 0.5
i_plot = 4

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

psfYrange = slice(psfcenter[0] - psfradius, psfcenter[0] + psfradius + 1)
psfXrange = slice(psfcenter[1] - psfradius, psfcenter[1] + psfradius + 1)

psfflux = flux[:,psfYrange, psfXrange].copy()
fill_flux = flux
psf_wl = wl[wl_ix]

plt.ioff()
plt.imshow(fill_flux[i_plot])
plt.colorbar()
plt.show()


fill_flux[:, psfYrange, psfXrange] = np.ma.masked

plt.ioff()
plt.imshow(fill_flux[i_plot])
plt.colorbar()
plt.show()

for i in xrange(fill_flux.shape[0]):
    print 'Filling %d of %d' % (i, fill_flux.shape[0])
    pa, ba = getEllipseParams(fill_flux[i], galaxycenter[1], galaxycenter[0])
    fill_flux[i] = fillImage(fill_flux[i], galaxycenter[1], galaxycenter[0], pa, ba)

psfbg = fill_flux[:, psfYrange, psfXrange]


plt.ioff()
plt.imshow(fill_flux[i_plot])
plt.colorbar()
plt.show()

plt.ioff()
plt.imshow(psfflux[i_plot])
plt.colorbar()
plt.show()

plt.ioff()
plt.imshow(psfbg[i_plot])
plt.colorbar()
plt.show()

plt.ioff()
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

plt.plot(bins_c, flux_r)
plt.plot(bins_c, flux_bg_r)
plt.plot(bins_c, bg_r)
plt.show()

moffat = function_description('Moffat')
moffat.PA.setValue(60, (0, 180))
moffat.ell.setValue(0.13, (0.0, 0.25))
moffat.I_0.setValue(1.0, (1e-20, 10.0))
moffat.fwhm.setValue(2.0, (1e-20, 5.0))
moffat.beta.setValue(1.1, (1e-20, 20.0))
model = SimpleModelDescription()
model.x0.setValue(psfradius)
model.x0.setTolerance(0.2)
model.y0.setValue(psfradius)
model.y0.setTolerance(0.2)
model.addFunction(moffat)

psfmodels = []
psfflags = []
for i in xrange(psfflux.shape[0]):
    print 'Fitting PSF %d of %d' % (i, fill_flux.shape[0])
    imfit = Imfit(model, quiet=True)
    imfit.fit(psfflux[i] - psfbg[i], mode='NM', use_model_for_errors=True)
    psfflags.append(not imfit.fitConverged)
    print '    Fit statistic: %f' % imfit.fitStatistic
    psfmodels.append(imfit.getModelDescription())


plt.clf()
plt.figure(figsize=(8, 11))
plt.subplot(411)
fwhm = np.ma.array([m.Moffat.fwhm.value for m in psfmodels], mask=psfflags)
plt.plot(psf_wl, fwhm, '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'FWHM $[arcsec]$')
plt.ylim(0.0, 3.5)
plt.xlim(psf_wl.min(), psf_wl.max())
fwhm_median = np.median(fwhm)
plt.hlines(fwhm_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, fwhm_median + 0.075, '%.3f' % fwhm_median)

plt.subplot(412)
beta = np.ma.array([m.Moffat.beta.value for m in psfmodels], mask=psfflags)
plt.plot(psf_wl, beta, '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'$\beta$')
plt.ylim(0.0, 1.5)
plt.xlim(psf_wl.min(), psf_wl.max())
beta_median = np.median(beta)
plt.hlines(beta_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, beta_median + 0.075, '%.3f' % beta_median)

plt.subplot(413)
PA = []
ell = []
for _PA, _ell in zip([m.Moffat.PA.value for m in psfmodels], [m.Moffat.ell.value for m in psfmodels]):
    _PA, _ell = fix_PA_ell(_PA, _ell)
    PA.append(_PA)
    ell.append(_ell)
PA = np.ma.array(PA, mask=psfflags)
ell = np.ma.array(ell, mask=psfflags)
plt.plot(psf_wl, ell, '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'$\epsilon$')
#plt.ylim(0.0, 1.5)
plt.xlim(psf_wl.min(), psf_wl.max())
ell_median = np.median(ell)
plt.hlines(ell_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, ell_median, '%.3f' % ell_median)

plt.subplot(414)
plt.plot(psf_wl, PA, '-k')
plt.xlabel(r'wavelength $[\AA]$')
plt.ylabel(r'P.A. $[degrees]$')
#plt.ylim(0.0, 1.5)
plt.xlim(psf_wl.min(), psf_wl.max())
PA_median = np.median(PA)
plt.hlines(PA_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, PA_median, '%.1f' % PA_median)

plt.suptitle('PSF Moffat parameters for %s (%s)' % (galaxy, cubeversion))
plt.savefig(cubeversion + '.PSF.png')

