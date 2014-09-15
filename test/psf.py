'''
Created on 10/09/2014

@author: andre
'''

import pyfits
import numpy as np
import matplotlib.pyplot as plt
from pycasso.util import radialProfile, fillImage, getEllipseParams

cube = '../data/NGC2916.V500.rscube.fits.gz'

f = pyfits.open(cube)
mask = f[3].data
flux = np.ma.array(f[0].data, mask=mask)
err = np.ma.array(f[1].data, mask=mask)

lambda0 = f[0].header['CRVAL3']
dlambda = f[0].header['CDELT3']
wl = np.arange(lambda0, lambda0 + (flux.shape[0]) * dlambda, dlambda)
f.close()

galaxycenter = (32.6, 35.3)

psfcenter = (45, 32)
psfradius = 7
psfYrange = slice(psfcenter[0] - psfradius, psfcenter[0] + psfradius + 1)
psfXrange = slice(psfcenter[1] - psfradius, psfcenter[1] + psfradius + 1)
psfLrange = slice(0, -1, 20)
i_plot = 4

psfflux = flux[psfLrange, psfYrange, psfXrange].copy()
psferr = err[psfLrange, psfYrange, psfXrange].copy()
fill_flux = flux[psfLrange]
psf_wl = wl[psfLrange]

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

from imfit import Imfit, SimpleModelDescription, function_description
moffat = function_description('Moffat')
moffat.PA.setValue(90, (0, 180))
moffat.ell.setValue(0.0, (-0.2, 0.2))
moffat.I_0.setValue(1.0, (0.0, 3.0))
moffat.fwhm.setValue(5.0, (0.0, 10.0))
moffat.beta.setValue(1.0, (0.0, 20.0))
model = SimpleModelDescription()
model.x0.setValue(psfradius)
model.x0.setTolerance(0.2)
model.y0.setValue(psfradius)
model.y0.setTolerance(0.2)
model.addFunction(moffat)

psfmodels = []
for i in xrange(psfflux.shape[0]):
    print 'Fitting PSF %d of %d' % (i, fill_flux.shape[0])
    imfit = Imfit(model, quiet=True)
    imfit.fit(psfflux[i] - psfbg[i], psferr[i_plot], mode='NM')
    print '    Fit statistic: %f' % imfit.fitStatistic
    psfmodels.append(imfit.getModelDescription())

plt.plot(psf_wl, [m.Moffat.fwhm.value for m in psfmodels])
plt.show()

plt.plot(psf_wl, [m.Moffat.beta.value for m in psfmodels])
plt.show()

