'''
Created on 10/09/2014

@author: andre
'''

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from pycasso.util import radialProfile
from specmorph.geometry import fix_PA_ell, distance
from imfit import Imfit, SimpleModelDescription, function_description
import sys

star = sys.argv[1]
date = sys.argv[2]
grating = sys.argv[3]
func = sys.argv[4] # 'Kolmogorov', 'Moffat, 'Gaussian'
beta4 = len(sys.argv) > 5 and sys.argv[5] == 'beta4'
if beta4:
    name = '%sBeta4' % (func)
else:
    name = func

cube = '../../../cubes.calibration/%s.%s.%s.scube.fits' % (star, date, grating)
psfLstep = 400.0 # /AA
badpix_frac = 0.5
sigma2fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
debug = True

print 'Fitting %s profiles' % func

f = pyfits.open(cube)
h = f[0].header
lambda0 = h['CRVAL3']
dlambda = h['CDELT3']

_mask = f[3].data
_flux = np.ma.array(f[0].data, mask=_mask)
_err = np.ma.array(f[1].data, mask=_mask)
wl = np.arange(lambda0, lambda0 + (_flux.shape[0]) * dlambda, dlambda)

image = _flux.sum(axis=0)
y0, x0 = np.where(image == np.nanmax(image))
r = distance(image.shape, x0, y0)
r_mask = r > 30
max_valid_pix = (~r_mask).sum()

_flux[:, r_mask] = np.ma.masked
_err[:, r_mask] = np.ma.masked


print 'Estimated star center: ', y0, x0
f.close()

psfLstep_ix = int(np.ceil(psfLstep / dlambda))
wl_ix = np.arange(0, len(wl), psfLstep_ix)
psf_wl = wl[wl_ix]

flux = np.ma.empty((len(wl_ix), _flux.shape[1], _flux.shape[2]))
err = np.ma.empty((len(wl_ix), _flux.shape[1], _flux.shape[2]))


for i, j in enumerate(wl_ix):
    j1 = j
    j2 = j + psfLstep_ix
    if j2 >= len(wl): j2 = len(wl) - 1
    jc = (j1 + j2) / 2
    print 'Computing binned image %d of %d (@ %.1f \\AA)' % (i, len(wl_ix), wl[jc])
    max_npts = psfLstep_ix
    masked_pts = _flux.mask[j1:j2].sum(axis=0)
    mask = masked_pts > (badpix_frac * max_npts)
    flux[i] = np.ma.mean(_flux[j1:j2], axis=0)
    flux[i, mask] = np.ma.masked
    err[i] = np.sqrt(np.ma.sum(_err[j1:j2]**2, axis=0))
    err[i, mask] = np.ma.masked

flux[flux <= 0.0] = np.ma.masked

fmax = flux.max()
fnorm = fmax
flux /= fnorm
err /= fnorm

psf_func = function_description(func, 'psf')
psf_func.PA.setValue(0.0, vmin=-190, vmax=190)
psf_func.ell.setValue(0.2, vmin=-1.0, vmax=1.0)
psf_func.I_0.setValue(fmax / fnorm, vmin=1e-20, vmax=10.0 * fmax / fnorm)
if func == 'Gaussian':
    psf_func.sigma.setValue(1.0, vmin=1e-20, vmax=20.0)
else:
    psf_func.fwhm.setValue(3.0, vmin=1e-20, vmax=20.0)
if func == 'Moffat':
    if beta4:
        psf_func.beta.setValue(4.0, fixed=True)
    else:
        psf_func.beta.setValue(2.0, vmin=1.0, vmax=20.0)

model = SimpleModelDescription()
model.x0.setValue(x0)
model.x0.setLimitsRel(5, 5)
model.y0.setValue(y0)
model.y0.setLimitsRel(5, 5)
model.addFunction(psf_func)

print 'Initial PSF model:'
print model

psfmodels = []
psfflags = []
goodfraction = []
chi2 = []
for i in xrange(flux.shape[0]):
    wl = psf_wl[i]
    print 'Fitting PSF %d of %d (%d)' % (i, flux.shape[0], wl)
    imfit = Imfit(model, quiet=True)
    imfit.fit(flux[i], mode='NM',
              use_model_for_errors=False)
    _goodfraction = float(imfit.nValidPixels) / float(max_valid_pix)
    goodfraction.append(_goodfraction)
    flagged = not imfit.fitConverged or _goodfraction < 0.7
    psfflags.append(flagged)
    print '    Fit statistic: %f' % imfit.fitStatistic
    print '    Reduced fit statistic: %f' % imfit.reducedFitStatistic
    print '    Valid pix: %.1f %%' % (_goodfraction * 100.0)
    print '    Flagged? %s' % flagged
    
    chi2.append(imfit.reducedFitStatistic)
    fitmodel = imfit.getModelDescription()
    modelimage = imfit.getModelImage()
    psfmodels.append(fitmodel)
    print fitmodel
    if debug:
        norm = flux[i].max()
        imradius = 10
        x0 = int(fitmodel.x0.value)
        y0 = int(fitmodel.y0.value)
        xslice = slice(x0 - imradius, x0 + imradius)
        yslice = slice(y0 - imradius, y0 + imradius)
        residual = flux[i] - modelimage
    
        plt.ioff()
        plt.clf()
        fig = plt.figure(1, figsize=(10, 7))
        gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
        ax = plt.subplot(gs[0,0])
        im = ax.imshow(flux[i, yslice, xslice] / norm, vmin=0.0, vmax=1.1, cmap='OrRd')
        plt.colorbar(im, ax=ax)
        ax.set_title('original')
        ax = plt.subplot(gs[0,1])
        im = ax.imshow(modelimage[yslice, xslice] / norm, vmin=0.0, vmax=1.1, cmap='OrRd')
        plt.colorbar(im, ax=ax)
        ax.set_title('model')
        
        ax = plt.subplot(gs[0,2])
        #res_max = max(np.abs(np.min(residual)), np.max(residual))
        im = ax.imshow(residual[yslice, xslice] / norm, vmin=-0.35, vmax=0.35, cmap='RdBu')
        plt.colorbar(im, ax=ax)
        ax.set_title('residual')
        
        bins = np.arange(0, imradius)
        bins_c = bins[:-1]
        pa = fitmodel.psf.PA.value
        ell = fitmodel.psf.ell.value
        pa, ell = fix_PA_ell(pa, ell)
        pa = np.pi * (pa + 90) / 180.0
        ba = 1.0 - ell
        x0 = fitmodel.x0.value - 1
        y0 = fitmodel.y0.value - 1
        f_r = radialProfile(flux[i] / norm, bins, x0, y0, pa, ba)
        err_r = radialProfile(flux[i] / norm, bins, x0, y0, pa, ba, mode='std')
        model_f_r = radialProfile(modelimage / norm, bins, x0, y0, pa, ba)
        residual_r = radialProfile(residual / norm, bins, x0, y0, pa, ba)
         
        if func == 'Gaussian':
            fwhm = sigma2fwhm * fitmodel.psf.sigma.value
        else:
            fwhm = fitmodel.psf.fwhm.value
    
        ax = plt.subplot(gs[1,:])
        ax.errorbar(bins_c, f_r, err_r, linestyle='-', color='k', ecolor='k', label='original')
        ax.plot(bins_c, model_f_r, 'k--', label='model')
        ax.plot(bins_c, residual_r, 'k:', label='residual')
        ax.vlines(fwhm / 2, -0.1, 1.2, linestyles='dashdot')
        ax.text(fwhm / 2 + 0.1, f_r.max() / 2 + 0.1, 'FWHM = %.3f "' % fwhm)
        ax.legend(loc='upper right')
        ax.set_ylim(-0.1, 1.2)
        ax.set_xlim(-0.1, bins_c.max() + 0.1)

        if beta4:
            plt.suptitle(r'%s PSF fit with $\beta=4$ for %s @ %s (%s, $%d\,\AA$)' % (func, star, date, grating, wl))
        else:
            plt.suptitle(r'%s PSF fit for %s @ %s (%s, $%d\,\AA$)' % (func, star, date, grating, wl))
    
        gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
        plt.show()

params = np.ma.empty(len(psfmodels), dtype=[('lambda', 'float64'), ('I_0', 'float64'),
                                            ('fwhm', 'float64'), ('beta', 'float64'), 
                                            ('x0', 'float64'), ('y0', 'float64'),
                                            ('PA', 'float64'), ('ell', 'float64'),
                                            ('good', 'float64'), ('flag', 'bool'), ('chi2', 'float64')])

psfflags = np.array(psfflags)
params['lambda'] = psf_wl
params['I_0'] = [m.psf.I_0.value for m in psfmodels]
if func == 'Gaussian':
    params['fwhm'] = [sigma2fwhm * m.psf.sigma.value for m in psfmodels]
else:
    params['fwhm'] = [m.psf.fwhm.value for m in psfmodels]
if func == 'Moffat':
    params['beta'] = [m.psf.beta.value for m in psfmodels]
else:
    params['beta'] = [0.0 for m in psfmodels]
params['x0'] = [m.x0.value for m in psfmodels]
params['y0'] = [m.y0.value for m in psfmodels]
PA = []
ell = []
for _PA, _ell in zip([m.psf.PA.value for m in psfmodels], [m.psf.ell.value for m in psfmodels]):
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
np.savetxt('out_calib/%s_%s.%s.%s.v1.5.PSF.dat' % (name, star, date, grating), params, header=header)


def getstats(p, wei):
    p_wei = np.sum(p * wei)
    p_var = np.sum((p - p_wei)**2 * wei)
    p_std = np.sqrt(p_var)
    return p_wei, p_std

wei = np.exp(-0.5 * params['chi2'] / params['chi2'].min())
#wei = np.ones_like(params['fwhm'])
wei /= np.sum(wei)
fwhm_wei, fwhm_std = getstats(params['fwhm'], wei)
beta_wei, beta_std = getstats(params['beta'], wei)
ell_wei, ell_std = getstats(params['ell'], wei)
x0_wei, x0_std = getstats(params['x0'], wei)
y0_wei, y0_std = getstats(params['y0'], wei)

print 'FWHM = %.3f +- %.3f' % (fwhm_wei, fwhm_std)
if func == 'Moffat' and not beta4:
    print 'beta = %.3f +- %.3f' % (beta_wei, beta_std)
print 'ell = %.3f +- %.3f' % (ell_wei, ell_std)
print 'x0 = %.3f +- %.3f' % (x0_wei, x0_std)
print 'y0 = %.3f +- %.3f' % (y0_wei, y0_std)

plt.clf()
plt.figure(figsize=(8, 12))
plt.subplot(511)
plt.plot(psf_wl, params['fwhm'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'FWHM $[arcsec]$')
#plt.ylim(0.0, 5.0)
plt.xlim(psf_wl.min(), psf_wl.max())
plt.hlines(fwhm_wei, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.hlines(fwhm_wei + fwhm_std, psf_wl.min(), psf_wl.max(), linestyles='dotted', colors='k')
plt.hlines(fwhm_wei - fwhm_std, psf_wl.min(), psf_wl.max(), linestyles='dotted', colors='k')
plt.text(psf_wl.min() + 100, fwhm_wei + 0.075, '%.3f' % fwhm_wei)

if func == 'Moffat' and not beta4:
    plt.subplot(512)
    plt.plot(psf_wl, params['beta'], '-k')
    #plt.xlabel(r'wavelength $[\AA]$')
    plt.gca().set_xticklabels([])
    plt.ylabel(r'$\beta$')
    #plt.ylim(0.0, 4.0)
    plt.xlim(psf_wl.min(), psf_wl.max())
    plt.hlines(beta_wei, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
    plt.hlines(beta_wei + beta_std, psf_wl.min(), psf_wl.max(), linestyles='dotted', colors='k')
    plt.hlines(beta_wei - beta_std, psf_wl.min(), psf_wl.max(), linestyles='dotted', colors='k')
    plt.text(psf_wl.min() + 100, beta_wei + 0.075, '%.3f' % beta_wei)

plt.subplot(513)
plt.plot(psf_wl, params['ell'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'$\epsilon$')
plt.ylim(0.0, 1.05)
plt.xlim(psf_wl.min(), psf_wl.max())
plt.hlines(ell_wei, psf_wl.min(), psf_wl.max(), linestyles='dashed', colors='k')
plt.hlines(ell_wei + ell_std, psf_wl.min(), psf_wl.max(), linestyles='dotted', colors='k')
plt.hlines(ell_wei - ell_std, psf_wl.min(), psf_wl.max(), linestyles='dotted', colors='k')
plt.text(psf_wl.min() + 100, ell_wei, '%.3f' % ell_wei)
plt.subplot(514)
plt.plot(psf_wl, params['PA'], '-k')
#plt.xlabel(r'wavelength $[\AA]$')
plt.gca().set_xticklabels([])
plt.ylabel(r'P.A. $[degrees]$')
plt.ylim(0.0, 180.0)
plt.xlim(psf_wl.min(), psf_wl.max())

plt.subplot(515)
plt.plot(psf_wl, params['chi2'], '-k')
plt.xlabel(r'wavelength $[\AA]$')
plt.ylabel(r'$\chi^2$')
plt.xlim(psf_wl.min(), psf_wl.max())

if beta4:
    plt.suptitle(r'%s PSF parameters with $\beta=4$ for %s @ %s (%s)' % (func, star, date, grating))
else:
    plt.suptitle('%s PSF parameters for %s @ %s (%s)' % (func, star, date, grating))
plt.savefig('out_calib/%s_%s.%s.%s.v1.5.PSF.png' % (name, star, date, grating))

