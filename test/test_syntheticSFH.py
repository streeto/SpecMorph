'''
Created on 30/05/2014

@author: andre
'''


from specmorph.components import SyntheticSFH
from specmorph.model import BDModel, bd_initial_model
from specmorph.fitting import fit_image
from specmorph.decomposition import IFSDecomposer
from specmorph.geometry import distance
from specmorph.util import logger, find_nearest_index

from pystarlight.util.base import StarlightBase
from pycasso.util import radialProfile
from imfit import Imfit, gaussian_psf
import numpy as np
import matplotlib.pyplot as plt
import time
from os import path

logger.setLevel(-1)

################################################################################
def create_image(shape, PSF, model):
    imfit = Imfit(model, PSF, quiet=True)
    image = imfit.getModelImage(shape)
    return image
################################################################################


################################################################################
def create_model_images(shape, PSF, model):
    bulge_model = model.getBulge()
    disk_model = model.getDisk()
    
    bulge = create_image(shape, PSF, bulge_model)
    disk = create_image(shape, PSF, disk_model)
    return bulge, disk
################################################################################

    
################################################################################
def smooth_param_polynomial(param, wl, flags, l_obs, degree=1):
    flag_ok = (flags == 0) & (wl > 4500.0)
    from astropy.modeling import models, fitting
    line = models.Polynomial1D(degree)
    fit = fitting.LinearLSQFitter()
    param_fitted = fit(line, wl[flag_ok], param[flag_ok])
    return param_fitted(l_obs)
################################################################################


################################################################################
def smooth_models(models, wl):
    params = np.array([m.getParams() for m in models], dtype=models[0].dtype)
    smooth_params = np.empty(len(wl), dtype=params[0].dtype)    
    param_wl = params['wl']
    param_flag = params['flag']

    for p in params.dtype.names:
        if p in ['wl', 'flag', 'chi2', 'n_pix']: continue
        smooth_params[p] = smooth_param_polynomial(params[p], param_wl, param_flag, wl, degree=1)
    
    models = []
    for i in xrange(len(smooth_params)):
        m = BDModel.fromParamVector(smooth_params[i])
        m.x0.fixed=True
        m.y0.fixed=True
        m.bulge.r_e.fixed=True
        m.bulge.n.fixed=True
        m.bulge.PA.fixed=True
        m.bulge.ell.fixed=True
        m.disk.h.fixed=True
        m.disk.PA.fixed=True
        m.disk.ell.fixed=True
        models.append(m)
    return models
################################################################################
    
debug = True
basedir = '../data/starlight/BasesDir'
basefile = '../data/starlight/BASE.gsd6e'
logger.info('Loading base %s', path.basename(basefile))
t1 = time.clock()
base = StarlightBase(basefile, basedir)
logger.info('Took %.2f seconds to read the base (%d files)' % (time.clock() - t1, base.sspfile.size))
wl_norm_window = (base.l_ssp < 5680.0) & (base.l_ssp > 5590.0)

logger.info('Computing synthetic SFH and their spectra.')
bulge_sfh = SyntheticSFH(base.ageBase)
bulge_sfh.addExp(14e9, 0.5e9, 1.0)
bulge_flux = (base.f_ssp * bulge_sfh.massVector()[:, np.newaxis]).sum(axis=1).sum(axis=0)
bulge_flux /= np.median(bulge_flux[wl_norm_window])

disk_sfh = SyntheticSFH(base.ageBase)
disk_sfh.addSquare(0.0, 5e9, 1.0)
disk_flux = (base.f_ssp * disk_sfh.massVector()[:, np.newaxis]).sum(axis=1).sum(axis=0)
disk_flux /= np.median(disk_flux[wl_norm_window])

if debug:
    plt.ioff()
    plt.plot(base.ageBase, bulge_sfh.massVector())
    plt.plot(base.ageBase, disk_sfh.massVector())
    plt.show()

if debug:
    plt.ioff()
    plt.plot(base.l_ssp, bulge_flux)
    plt.plot(base.l_ssp, disk_flux)
    plt.show()

logger.info('Creating true model.')
flux_unit = 1e-16
shape = (72,77)
x0 = 36.0
y0 = 33.0
ba = 0.9
ell = 1.0 - ba
pa = 45.0
pa_rad = pa / 180 * np.pi
flagged = distance(shape, x0, y0) > 32.0
noise = 0.05

true_model = BDModel(0, x0, y0,
                     I_e=2, r_e=5, n=2.0, PA_b=pa, ell_b=ell,
                     I_0=1, h=15, PA_d=pa, ell_d=ell)
PSF = gaussian_psf(2.4, size=9)
bulge_image, disk_image = create_model_images(shape, PSF, true_model)
bulge_image = np.ma.masked_where(flagged, bulge_image)
disk_image = np.ma.array(disk_image, mask=flagged)
model_image = bulge_image + disk_image
# bulge_frac = bulge_image / model_image
# disk_frac = disk_image / model_image
model_noise = model_image * noise

logger.info('Creating IFS.')
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

logger.info('Beginning decomposition.')
decomp = IFSDecomposer()
decomp.setSynthPSF(FWHM=2.4, size=9)
decomp.loadData(base.l_ssp, full_spectra, full_noise, np.zeros_like(full_spectra, dtype='bool'))

swll, swlu = 5590.0, 5680.0
sl1 = find_nearest_index(decomp.wl, swll)
sl2 = find_nearest_index(decomp.wl, swlu)
qSignal, qNoise, qWl = decomp.getSpectraSlice(sl1, sl2)

if debug:
    plt.ioff()
    plt.clf()
    plt.imshow(qSignal)
    plt.colorbar()
    plt.show()
    
    bins = np.arange(0, 32)
    bins_c = bins[:-1] + 0.5
    bulgeim, diskim = create_model_images(qSignal.shape, PSF, true_model)
    qsr = radialProfile(np.log10(qSignal), bins, x0, y0, pa, ba)
    br = radialProfile(np.log10(bulgeim), bins, x0, y0, pa, ba)
    dr = radialProfile(np.log10(diskim), bins, x0, y0, pa, ba)
    plt.ioff()
    plt.clf()
    plt.plot(bins_c, qsr, 'k-')
    plt.plot(bins_c, br, 'r-')
    plt.plot(bins_c, dr, 'b-')
    plt.show()

logger.warn('Computing initial model (takes a LOT of time).')
t1 = time.time()
initial_model = bd_initial_model(qSignal, qNoise, PSF, x0, y0)
logger.info('Refined initial model:\n%s\n' % initial_model)
logger.warn('Initial model time: %.2f\n' % (time.time() - t1))

if debug:
    bins = np.arange(0, 32)
    bins_c = bins[:-1] + 0.5
    bulgeim, diskim = create_model_images(qSignal.shape, PSF, initial_model)
    qsr = radialProfile(np.log10(qSignal), bins, x0, y0, pa, ba)
    br = radialProfile(np.log10(bulgeim), bins, x0, y0, pa, ba)
    dr = radialProfile(np.log10(diskim), bins, x0, y0, pa, ba)
    plt.ioff()
    plt.clf()
    plt.plot(bins_c, qsr, 'k-')
    plt.plot(bins_c, br, 'r-')
    plt.plot(bins_c, dr, 'b-')
    plt.show()

logger.info('Starting first pass modeling.')
t1 = time.time()
first_pass_models = decomp.fitSpectra(step=100, box_radius=50, initial_model=initial_model, mode='NM')
first_pass_params = np.array([m.getParams() for m in first_pass_models], dtype=first_pass_models[0].dtype)
first_pass_lambdas = decomp.wl[::100]
logger.info('Done first pass modeling, time: %.2f' % (time.time() - t1))

logger.info('Smoothing parameters.')
smoothed_models = smooth_models(first_pass_models, decomp.wl)
smoothed_params = np.array([m.getParams() for m in smoothed_models], dtype=smoothed_models[0].dtype)
        
logger.info('Starting second pass modeling...')
t1 = time.time()
models = decomp.fitSpectra(step=1, box_radius=0, initial_model=smoothed_models, mode='LM', insist=True)
fitted_params = np.array([m.getParams() for m in models], dtype=models[0].dtype)
logger.info('Done second pass modeling, time: %.2f' % (time.time() - t1))





