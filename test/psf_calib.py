'''
Created on 10/09/2014

@author: andre
'''

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from pycasso.util import radialProfile, getEllipseParams
from specmorph.geometry import fix_PA_ell
from imfit import Imfit, SimpleModelDescription, function_description
import sys

star = sys.argv[1]
date = sys.argv[2]
grating = sys.argv[3]
cube = '../../cubes.calibration/%s.%s.%s.scube.fits' % (star, date, grating)
psfradius = 7
psfLstep = 40.0 # /AA
badpix_frac = 0.5

debug = True
i_plot = 3

f = pyfits.open(cube)
lambda0 = f[0].header['CRVAL3']
dlambda = f[0].header['CDELT3']
_mask = f[3].data
_flux = np.ma.array(f[0].data, mask=_mask)
wl = np.arange(lambda0, lambda0 + (_flux.shape[0]) * dlambda, dlambda)
image = _flux.sum(axis=0)
y0, x0 = np.where(image == np.nanmax(image))
print 'Galaxy center: ', y0, x0
f.close()

if debug:
    plt.clf()
    plt.ioff()
    plt.imshow(_flux[i_plot])
    plt.colorbar()
    plt.show()

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
    flux[i] = np.ma.sum(_flux[j1:j2], axis=0)
    
flux /= flux.max()

psf_wl = wl[wl_ix]


if debug:
    plt.clf()
    plt.ioff()
    plt.imshow(flux[i_plot])
    plt.colorbar()
    plt.show()

    bins = np.arange(0, 10)
    bins_c = bins[:-1] + 0.5
    pa = 90.0
    ba = 1.0
    flux_r = radialProfile(flux[i_plot], bins, x0, y0, pa, ba)
    
    plt.clf()
    plt.plot(bins_c, flux_r)
    plt.show()

moffat = function_description('Moffat')
moffat.PA.setValue(60, vmin=-190, vmax=190)
moffat.ell.setValue(0.2, vmin=-0.4, vmax=0.4)
moffat.I_0.setValue(1.0, vmin=1e-20, vmax=10.0)
moffat.fwhm.setValue(2.0, vmin=1e-20, vmax=20.0)
moffat.beta.setValue(1.1, vmin=1e-20, vmax=20.0)
model = SimpleModelDescription()
model.x0.setValue(x0)
model.x0.setLimitsRel(15, 15)
model.y0.setValue(y0)
model.y0.setLimitsRel(15, 15)
model.addFunction(moffat)

print 'Initial PSF model:'
print model

psfmodels = []
psfflags = []
goodfraction = []
chi2 = []
for i in xrange(flux.shape[0]):
    print 'Fitting PSF %d of %d' % (i, flux.shape[0])
    imfit = Imfit(model, quiet=True)
    imfit.fit(flux[i], mode='NM', use_model_for_errors=True)
    _goodfraction = float(imfit.nValidPixels) / float(flux[i].size)
    goodfraction.append(_goodfraction)
    psfflags.append(not imfit.fitConverged)
    print '    Reduced fit statistic: %f' % imfit.reducedFitStatistic
    chi2.append(imfit.reducedFitStatistic)
    fitmodel = imfit.getModelDescription()
    modelimage = imfit.getModelImage()
    psfmodels.append(fitmodel)
    if debug:
        print fitmodel
        bins = np.arange(0, 10)
        bins_c = bins[:-1] + 0.5
        pa = fitmodel.Moffat.PA.value
        ell = fitmodel.Moffat.ell.value
        pa, ell = fix_PA_ell(pa, ell)
        pa = np.pi * pa / 180.0
        ba = 1.0 - ell
        flux_r = radialProfile(flux[i_plot], bins, fitmodel.x0.value, fitmodel.y0.value,
                               fitmodel.Moffat.PA.value, 1.0 - fitmodel.Moffat.ell.value)
        flux_model_r = radialProfile(flux[i_plot], bins, fitmodel.x0.value, fitmodel.y0.value,
                                     pa, ba)
        
        plt.clf()
        plt.plot(bins_c, flux_r)
        plt.plot(bins_c, flux_model_r)
        plt.show()


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
np.savetxt('psf_calib/%s.%s.%s.v1.5.PSF.dat' % (star, date, grating), params, header=header)

plt.clf()
plt.figure(figsize=(8, 12))
plt.subplot(511)
plt.plot(psf_wl, params['fwhm'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'FWHM $[arcsec]$')
#plt.ylim(0.0, 5.0)
plt.xlim(psf_wl.min(), psf_wl.max())
fwhm_median = np.median(params['fwhm'])
plt.hlines(fwhm_median, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.text(psf_wl.min() + 100, fwhm_median + 0.075, '%.3f' % fwhm_median)

plt.subplot(512)
plt.plot(psf_wl, params['beta'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'$\beta$')
#plt.ylim(0.0, 4.0)
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

plt.suptitle('PSF Moffat parameters for %s @ %s (%s)' % (star, date, grating))
plt.savefig('psf_calib/%s.%s.%s.v1.5.PSF.png' % (star, date, grating))

