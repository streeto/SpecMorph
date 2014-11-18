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
grating = sys.argv[2]
func = sys.argv[3] # 'Kolmogorov', 'Moffat, 'Gaussian'
beta4 = len(sys.argv) > 4 and sys.argv[4] == 'beta4'
if beta4:
    name = '%sBeta4' % (func)
else:
    name = func

cube = '../../../cubes.DR2/%s.%s.rscube.fits.gz' % (galaxy, grating)
psfradius = 7
psfLstep = 400.0 # /AA
badpix_frac = 0.5
sigma2fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))

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
psf_wl = np.ma.empty((len(wl_ix)))
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
    psf_wl[i] = wl[jc]

psfYrange = slice(psfcenter[galaxy][0] - psfradius, psfcenter[galaxy][0] + psfradius + 1)
psfXrange = slice(psfcenter[galaxy][1] - psfradius, psfcenter[galaxy][1] + psfradius + 1)

psfflux = flux[:,psfYrange, psfXrange].copy()
fill_flux = flux

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

psf_func = function_description(func, 'psf')
psf_func.PA.setValue(0.0, vmin=-190, vmax=190)
psf_func.ell.setValue(0.2, vmin=-1.0, vmax=1.0)
psf_func.I_0.setValue(1.0, vmin=1e-20, vmax=10.0)
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
model.x0.setValue(psfradius)
model.x0.setLimitsRel(3, 3)
model.y0.setValue(psfradius)
model.y0.setLimitsRel(3, 3)
model.addFunction(psf_func)

print 'Initial PSF model:'
print model

psfmodels = []
psfflags = []
goodfraction = []
chi2 = []
for i in xrange(psfflux.shape[0]):
    wl = psf_wl[i]
    pf = psfflux[i] - psfbg[i]
    pf[pf <= 0] = np.ma.masked
    print 'Fitting PSF %d of %d (%s \AA)' % (i, fill_flux.shape[0], wl)
    imfit = Imfit(model, quiet=True)
    imfit.fit(pf, mode='NM', use_model_for_errors=False)
    _goodfraction = float(imfit.nValidPixels) / float(psfflux[i].size)
    goodfraction.append(_goodfraction)
    flagged = not imfit.fitConverged or _goodfraction < 0.5
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
        pf = psfflux[i] - psfbg[i]
        norm = pf.max()
        residual = pf - modelimage
    
        plt.ioff()
        plt.clf()
        fig = plt.figure(1, figsize=(10, 7))
        gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
        ax = plt.subplot(gs[0,0])
        im = ax.imshow(pf / norm, vmin=0.0, vmax=1.1, cmap='OrRd')
        plt.colorbar(im, ax=ax)
        ax.set_title('original')
        ax = plt.subplot(gs[0,1])
        im = ax.imshow(modelimage / norm, vmin=0.0, vmax=1.1, cmap='OrRd')
        plt.colorbar(im, ax=ax)
        ax.set_title('model')
        
        ax = plt.subplot(gs[0,2])
        #res_max = max(np.abs(np.min(residual)), np.max(residual))
        im = ax.imshow(residual / norm, vmin=-0.35, vmax=0.35, cmap='RdBu')
        plt.colorbar(im, ax=ax)
        ax.set_title('residual')
        
        bins = np.arange(0, psfradius)
        bins_c = bins[:-1]
        pa = fitmodel.psf.PA.value
        ell = fitmodel.psf.ell.value
        pa, ell = fix_PA_ell(pa, ell)
        pa = np.pi * (pa + 90) / 180.0
        ba = 1.0 - ell
        x0 = fitmodel.x0.value - 1
        y0 = fitmodel.y0.value - 1
        f_r = radialProfile(pf / norm, bins, x0, y0, pa, ba)
        err_r = radialProfile(pf / norm, bins, x0, y0, pa, ba, mode='std')
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
            plt.suptitle(r'%s PSF fit with $\beta=4$ for %s (%s, $%d\,\AA$)' % (func, galaxy, grating, wl))
        else:
            plt.suptitle(r'%s PSF fit for %s (%s, $%d\,\AA$)' % (func, galaxy, grating, wl))
    
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
np.savetxt('out/%s_%s.%s.v1.5.PSF.dat' % (name, galaxy, grating), params, header=header)

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
    plt.suptitle(r'%s PSF parameters with $\beta=4$ for %s (%s)' % (func, galaxy, grating))
else:
    plt.suptitle('%s PSF parameters for %s (%s)' % (func, galaxy, grating))
plt.savefig('out/%s_%s.%s.v1.5.PSF.png' % (name, galaxy, grating))

