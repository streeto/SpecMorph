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
from glob import glob
from os import path

cubes = glob('../../../images.calibration/*.g.fits')
psfLstep = 40.0 # /AA
badpix_frac = 0.5
fitradius = 30
debug = False

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

moffat = function_description('Moffat')
moffat.PA.setValue(60, vmin=-190, vmax=190)
moffat.ell.setValue(0.2, vmin=-1.0, vmax=1.0)
moffat.I_0.setValue(1.0, vmin=1e-20, vmax=10.0)
moffat.fwhm.setValue(2.5, vmin=1e-20, vmax=20.0)
moffat.beta.setValue(1.9, vmin=1e-20, vmax=20.0)
model = SimpleModelDescription()
model.x0.setValue(flux[0].shape[1] / 2)
model.x0.setLimitsRel(fitradius, fitradius)
model.y0.setValue(flux[0].shape[0] / 2)
model.y0.setLimitsRel(fitradius, fitradius)
model.addFunction(moffat)

print 'Initial PSF model:'
print model
good = []
chi2 = []
flag = []
fitmodel = []
modelflux = []
for i, f in enumerate(flux):
    print 'Fitting PSF %d of %d' % (i, len(flux))
    imfit = Imfit(model, quiet=True)
    imfit.fit(f, mode='NM', use_model_for_errors=True, use_cash_statistics=False)
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

    if debug:
        print _fitmodel
        plt.ioff()
        plt.figure(1, figsize=(10, 3))
        plt.clf()
        plt.subplot(131)
        plt.imshow(f, vmin=0.0, vmax=1.1)
        plt.colorbar()
        plt.title('original')
        plt.subplot(132)
        plt.imshow(model_f, vmin=0.0, vmax=1.1)
        plt.colorbar()
        plt.title('model')
        plt.subplot(133)
        plt.imshow((f - model_f))
        plt.colorbar()
        plt.title('residual')
        
        bins = np.arange(0, 10)
        bins_c = bins[:-1] + 0.5
        pa = _fitmodel.Moffat.PA.value
        ell = _fitmodel.Moffat.ell.value
        pa, ell = fix_PA_ell(pa, ell)
        pa = np.pi * pa / 180.0
        ba = 1.0 - ell
        f_r = radialProfile(f, bins, _fitmodel.x0.value - 1, _fitmodel.y0.value - 1,
                               pa, ba)
        model_f_r = radialProfile(model_f, bins, _fitmodel.x0.value - 1, _fitmodel.y0.value - 1,
                                     pa, ba)
        residual_r = radialProfile(f - model_f, bins, _fitmodel.x0.value - 1, _fitmodel.y0.value - 1,
                                     pa, ba)
         
        plt.figure(2, figsize=(5, 4))
        plt.clf()
        plt.plot(bins_c, f_r, label='original')
        plt.plot(bins_c, model_f_r, label='model')
        plt.plot(bins_c, residual_r, label='residual')
        plt.legend(loc='upper right')
        plt.show()

param_dtype = [('star', 'S40'), ('I_0', 'float64'), 
               ('fwhm', 'float64'), ('beta', 'float64'), 
               ('x0', 'float64'), ('y0', 'float64'), 
               ('PA', 'float64'), ('ell', 'float64'), 
               ('good', 'float64'), ('flag', 'int'), ('chi2', 'float64')]

params = np.empty(len(fitmodel), dtype=param_dtype)

param_fmt = ['%s', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%f', '%d', '%f']
params['star'] = [path.basename(c) for c in cubes]
params['I_0'] = [m.Moffat.I_0.value for m in fitmodel]
params['fwhm'] = [m.Moffat.fwhm.value for m in fitmodel]
params['beta'] = [m.Moffat.beta.value for m in fitmodel]
params['x0'] = [m.x0.value for m in fitmodel]
params['y0'] = [m.y0.value for m in fitmodel]
PA = []
ell = []
for _PA, _ell in zip([m.Moffat.PA.value for m in fitmodel], [m.Moffat.ell.value for m in fitmodel]):
    _PA, _ell = fix_PA_ell(_PA, _ell)
    PA.append(_PA)
    ell.append(_ell)
params['PA'] = PA
params['ell'] = ell
params['good'] = good
params['flag'] = flag
params['chi2'] = chi2

header =' '.join(params.dtype.names)
np.savetxt('out_calib/fit_gband.dat', params, header=header, fmt=param_fmt)
