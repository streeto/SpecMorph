'''
Created on 10/09/2014

@author: andre
'''

import pyfits
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pycasso.util import radialProfileExact
from specmorph.geometry import fix_PA_ell, distance
from imfit import Imfit, SimpleModelDescription, function_description
from glob import glob
from os import path
import sys

func = sys.argv[1] # 'Kolmogorov', 'Moffat, 'Gaussian'
print 'Fitting %s profiles' % func

cubes = glob('../../../images.calibration/*.g.fits')
fitradius = 40
sigma2fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
debug = True

flux = []
for cube in cubes:
    f = pyfits.open(cube)
    shape = f[0].data.shape
    _mask = f[0].data <= 0
    _flux = np.ma.array(f[0].data, mask=_mask)
    y0, x0 = np.where(_flux == _flux.max())
    r = distance(shape, x0, y0)
    _flux[r > fitradius] = np.ma.masked
    y0 = shape[0] / 2
    _flux /= _flux.max()
    flux.append(_flux)
    f.close()

psf_func = function_description(func, name='psf')
psf_func.PA.setValue(60, vmin=-190, vmax=190)
psf_func.ell.setValue(0.2, vmin=-1.0, vmax=1.0)
psf_func.I_0.setValue(1.0, vmin=1e-20, vmax=10.0)
if func == 'Gaussian':
    psf_func.sigma.setValue(1.0, vmin=0.5, vmax=20.0)
else:
    psf_func.fwhm.setValue(2.5, vmin=1e-20, vmax=20.0)
if func == 'Moffat':
    psf_func.beta.setValue(1.9, vmin=1e-20, vmax=20.0)
model = SimpleModelDescription()
model.x0.setValue(flux[0].shape[1] / 2)
model.x0.setLimitsRel(fitradius, fitradius)
model.y0.setValue(flux[0].shape[0] / 2)
model.y0.setLimitsRel(fitradius, fitradius)
model.addFunction(psf_func)

print 'Initial PSF model:'
print model
good = []
chi2 = []
flag = []
fitmodel = []
modelflux = []
for i, f in enumerate(flux):
    star = path.basename(cubes[i])
    print 'Fitting PSF %d of %d (%s)' % (i, len(flux), star)
    imfit = Imfit(model, quiet=True)
    imfit.fit(f, mode='NM', use_model_for_errors=False, use_cash_statistics=False)
    _goodfraction = float(imfit.nValidPixels) / float(f.size)
    print '    Good pixels: %.1f %%' % (_goodfraction  * 100)
    good.append(_goodfraction)
    print '    Fit converged? %s' % imfit.fitConverged
    flag.append(not imfit.fitConverged)
    print '    Fit statistic: %f' % imfit.fitStatistic
    chi2.append(imfit.fitStatistic)

    _fitmodel = imfit.getModelDescription()
    fitmodel.append(_fitmodel)
    model_f = imfit.getModelImage()
    modelflux.append(model_f)

    imradius = 10
    x0 = int(_fitmodel.x0.value)
    y0 = int(_fitmodel.y0.value)
    xslice = slice(x0 - imradius, x0 + imradius)
    yslice = slice(y0 - imradius, y0 + imradius)

    plt.ioff()
    plt.clf()
    fig = plt.figure(1, figsize=(10, 7))
    gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
    ax = plt.subplot(gs[0,0])
    im = ax.imshow(f[yslice, xslice], vmin=0.0, vmax=1.1, cmap='OrRd')
    plt.colorbar(im, ax=ax)
    ax.set_title('original')
    ax = plt.subplot(gs[0,1])
    im = ax.imshow(model_f[yslice, xslice], vmin=0.0, vmax=1.1, cmap='OrRd')
    plt.colorbar(im, ax=ax)
    ax.set_title('model')
    
    ax = plt.subplot(gs[0,2])
    residual = f - model_f
    #res_max = max(np.abs(np.min(residual)), np.max(residual))
    im = ax.imshow(residual[yslice, xslice], vmin=-0.35, vmax=0.35, cmap='RdBu')
    plt.colorbar(im, ax=ax)
    ax.set_title('residual')
    
    bins = np.arange(0, imradius)
    bins_c = bins[:-1]
    pa = _fitmodel.psf.PA.value
    ell = _fitmodel.psf.ell.value
    pa, ell = fix_PA_ell(pa, ell)
    pa = np.pi * pa / 180.0
    ba = 1.0 - ell
    f_r = radialProfileExact(f, bins, _fitmodel.x0.value - 1, _fitmodel.y0.value - 1,
                           pa, ba)
    model_f_r = radialProfileExact(model_f, bins, _fitmodel.x0.value - 1, _fitmodel.y0.value - 1,
                                 pa, ba)
    residual_r = radialProfileExact(f - model_f, bins, _fitmodel.x0.value - 1, _fitmodel.y0.value - 1,
                                 pa, ba)
     
    if func == 'Gaussian':
        fwhm = sigma2fwhm * _fitmodel.psf.sigma.value
    else:
        fwhm = _fitmodel.psf.fwhm.value

    ax = plt.subplot(gs[1,:])
    ax.plot(bins_c, f_r, 'k-', label='original')
    ax.plot(bins_c, model_f_r, 'k--', label='model')
    ax.plot(bins_c, residual_r, 'k:', label='residual')
    ax.vlines(fwhm / 2, -0.1, 1.0, linestyles='dashdot')
    ax.text(fwhm / 2 + 0.1, f_r.max() / 2 + 0.1, 'FWHM = %.3f "' % fwhm)
    ax.legend(loc='upper right')
    ax.set_ylim(-0.1, 1.0)
    
    plt.suptitle('%s PSF fit (%s)' % (func, star))

    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    plt.savefig('out_calib/%s_PSF.%s.g.png'  % (func, star))
    if debug:
        print _fitmodel
        plt.show()

param_dtype = [('star', 'S40'), ('I_0', 'float64'), 
               ('fwhm', 'float64'), ('beta', 'float64'), 
               ('x0', 'float64'), ('y0', 'float64'), 
               ('PA', 'float64'), ('ell', 'float64'), 
               ('good', 'float64'), ('flag', 'int'), ('chi2', 'float64')]

params = np.empty(len(fitmodel), dtype=param_dtype)

param_fmt = ['%s', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%d', '%f']
params['star'] = [path.basename(c) for c in cubes]
params['I_0'] = [m.psf.I_0.value for m in fitmodel]
if func == 'Gaussian':
    params['fwhm'] = [sigma2fwhm * m.psf.sigma.value for m in fitmodel]
else:
    params['fwhm'] = [m.psf.fwhm.value for m in fitmodel]
if func == 'Moffat':
    params['beta'] = [m.psf.beta.value for m in fitmodel]
else:
    params['beta'] = [0.0 for m in fitmodel]
params['x0'] = [m.x0.value for m in fitmodel]
params['y0'] = [m.y0.value for m in fitmodel]
PA = []
ell = []
for _PA, _ell in zip([m.psf.PA.value for m in fitmodel], [m.psf.ell.value for m in fitmodel]):
    _PA, _ell = fix_PA_ell(_PA, _ell)
    PA.append(_PA)
    ell.append(_ell)
params['PA'] = PA
params['ell'] = ell
params['good'] = good
params['flag'] = flag
params['chi2'] = chi2

header =' '.join(params.dtype.names)
np.savetxt('out_calib/%s_fit_gband.dat' % func, params, header=header, fmt=param_fmt)
