'''
Created on 30/05/2014

@author: andre
'''


from specmorph.components import SyntheticSFH
from specmorph.geometry import distance
from specmorph.util import logger, find_nearest_index

import numpy as np
from pystarlight.util.base import StarlightBase
import matplotlib.pyplot as plt
from imfit import Imfit, SimpleModelDescription, function_description, gaussian_psf
from specmorph.decomposition import IFSDecomposer
import sys
from specmorph.fitting import fit_image
from specmorph.model import bd_initial_model

logger.setLevel(-1)

def bulge_function(I_e, r_e, n, PA, ell):
    bulge = function_description('Sersic', name='bulge')
    bulge.I_e.setValue(I_e, [1e-33, 10*I_e])
    bulge.r_e.setValue(r_e, [1e-33, 10*r_e])
    bulge.n.setValue(n, [0.5, 5.9])
    bulge.PA.setValue(PA, [0, 180])
    bulge.ell.setValue(ell, [0, 1])
    return bulge


def disk_function(I_0, h, PA, ell):
    disk = function_description('Exponential', name='disk')
    disk.I_0.setValue(I_0, [1e-33, 10*I_0])
    disk.h.setValue(h, [1e-33, 10*h])
    disk.PA.setValue(PA, [0, 180])
    disk.ell.setValue(ell, [0, 1])
    return disk


def galaxy_model(x0, y0, bulge=False, disk=False,
                 PA_b=0.0, ell_b=0.0, I_e=0.0, r_e=0.0, n=0.0, PA_d=0.0, ell_d=0.0, I_0=0.0, h=0.0):
    if not (bulge or disk):
        raise Exception('At least one of bulge or disk must be set.') 
    model = SimpleModelDescription()
    model.x0.setValue(x0, [x0-10, x0+10])
    model.y0.setValue(y0, [y0-10, y0+10])
    if bulge:
        bulge = bulge_function(I_e, r_e, n, PA_b, ell_b)
        model.addFunction(bulge)
    if disk:
        disk = disk_function(I_0, h, PA_d, ell_d)
        model.addFunction(disk)
    return model



def create_image(shape, PSF, model):
    imfit = Imfit(model, PSF, quiet=True)
    image = imfit.getModelImage(shape)
    return image


def create_model_images(shape, PSF, model):
    x0 = model.x0.value
    y0 = model.y0.value

    I_e = model.bulge.I_e.value
    r_e = model.bulge.r_e.value
    n = model.bulge.n.value
    PA_b = model.bulge.PA.value
    ell_b = model.bulge.ell.value
    
    I_0 = model.disk.I_0.value
    h = model.disk.h.value
    PA_d = model.disk.PA.value
    ell_d = model.disk.ell.value
    
    bulge_model = galaxy_model(x0, y0, bulge=True, I_e=I_e, r_e=r_e, n=n, PA_b=PA_b, ell_b=ell_b)
    disk_model = galaxy_model(x0, y0, disk=True, I_0=I_0, h=h, PA_d=PA_d, ell_d=ell_d)
    
    bulge = create_image(shape, PSF, bulge_model)
    disk = create_image(shape, PSF, disk_model)
    return bulge, disk
    
    
basedir = '../data/starlight/BasesDir'
basefile = '../data/starlight/BASE.gsd6e'
base = StarlightBase(basefile, basedir)
wl_norm_window = (base.l_ssp < 5680.0) & (base.l_ssp > 5590.0)

bulge_sfh = SyntheticSFH(base.ageBase)
bulge_sfh.addExp(14e9, 1e9, 1.0)
bulge_flux = (base.f_ssp * bulge_sfh.massVector()[:, np.newaxis, np.newaxis]).sum(axis=1).sum(axis=0)
bulge_flux /= np.median(bulge_flux[wl_norm_window])

disk_sfh = SyntheticSFH(base.ageBase)
disk_sfh.addSquare(14e9, 14e9, 1.0)
disk_flux = (base.f_ssp * disk_sfh.massVector()[:, np.newaxis, np.newaxis]).sum(axis=1).sum(axis=0)
disk_flux /= np.median(disk_flux[wl_norm_window])

#plt.ioff()
# plt.plot(base.l_ssp, bulge_flux)
# plt.plot(base.l_ssp, disk_flux)
# plt.show()

flux_unit = 1e-16
shape = (72,77)
x0 = 36.0
y0 = 33.0
ba = 0.7
ell = 1.0 - ba
pa = 45.0
pa_rad = pa / 180 * np.pi
flagged = distance(shape, x0, y0) > 32.0
noise = 0.05

true_model = galaxy_model(x0, y0, bulge=True, disk=True,
                     I_e=1, r_e=8, n=2.0, PA_b=pa, ell_b=ell,
                     I_0=1, h=12, PA_d=pa, ell_d=ell)
PSF = gaussian_psf(2.4, size=9)
bulge_image, disk_image = create_model_images(shape, PSF, true_model)
bulge_image = np.ma.masked_where(flagged, bulge_image)
disk_image = np.ma.array(disk_image, mask=flagged)
model_image = bulge_image + disk_image
# bulge_frac = bulge_image / model_image
# disk_frac = disk_image / model_image
model_noise = model_image * noise

bulge_spectra = bulge_image * (flux_unit * bulge_flux[..., np.newaxis, np.newaxis]) 
disk_spectra = disk_image * (flux_unit * disk_flux[..., np.newaxis, np.newaxis]) 
full_spectra = bulge_spectra + disk_spectra
full_noise = full_spectra * noise
# qNoise = np.ones_like(qSignal) * 0.1
# Add gaussian noise to spectra.
tmp_noise = np.zeros(full_spectra.shape)
for i in xrange(shape[0]):
    for j in xrange(shape[1]):
        tmp_noise[:, i,j] = np.random.normal(0.0, model_noise.data[i,j] * flux_unit, len(base.l_ssp))
full_spectra += tmp_noise

# plt.ioff()
# plt.imshow(full_spectra[1000])
# plt.colorbar()
# plt.show()

decomp = IFSDecomposer()
decomp.setSynthPSF(PSF_FWHM=2.4, PSF_size=9)
decomp.loadData(base.l_ssp, full_spectra, full_noise, np.zeros_like(full_spectra, dtype='bool'))

swll, swlu = 5590.0, 5680.0
sl1 = find_nearest_index(decomp.wl, swll)
sl2 = find_nearest_index(decomp.wl, swlu)
qSignal, qNoise, qWl = decomp.getSpectraSlice(sl1, sl2)

print qWl
plt.ioff()
plt.imshow(qSignal)
plt.colorbar()
plt.show()

guess_model = bd_initial_model(qSignal, x0, y0)
print guess_model
guess_model, converged, chi2 = fit_image(qSignal, qNoise, decomp.PSF, guess_model, mode='DE', quiet=False)
print 'chi2 = %.1f' % chi2
print guess_model


models = decomp.fitSpectra(step=100, box_radius=0, initial_model=guess_model, mode='LM', insist=True)




