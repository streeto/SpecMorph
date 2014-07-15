'''
Created on 30/05/2014

@author: andre
'''


from specmorph.components import SyntheticSFH
from specmorph.model import BDModel, bd_initial_model, create_model_images
from specmorph.decomposition import IFSDecomposer
from specmorph.geometry import distance, ellipse_params
from specmorph.util import logger, find_nearest_index

from pystarlight.util.base import StarlightBase
from pycasso.util import radialProfile
from imfit import gaussian_psf
import numpy as np
import matplotlib.pyplot as plt
import time
from os import path
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator

logger.setLevel(-1)

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
##########
########## Plot setup 
##########
################################################################################
pdf = PdfPages('test.pdf')
plotpars = {'legend.fontsize': 8,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'text.fontsize': 10,
            'axes.titlesize': 12,
            'font.family': 'Times New Roman',
#             'figure.subplot.left': 0.08,
#             'figure.subplot.bottom': 0.08,
#             'figure.subplot.right': 0.97,
#             'figure.subplot.top': 0.95,
#             'figure.subplot.wspace': 0.42,
#             'figure.subplot.hspace': 0.1,
            'image.cmap': 'GnBu',
            }
plt.rcParams.update(plotpars)
plt.ioff()

################################################################################
##########
########## Population model setup
##########
################################################################################

basedir = '../data/starlight/BasesDir'
basefile = '../data/starlight/BASE.gsd6e'
logger.info('Loading base %s', path.basename(basefile))
t1 = time.clock()
base = StarlightBase(basefile, basedir)
logger.info('Took %.2f seconds to read the base (%d files)' % (time.clock() - t1, base.sspfile.size))
wl_norm_window = (base.l_ssp < 5680.0) & (base.l_ssp > 5590.0)

logger.info('Computing synthetic SFH and their spectra.')
bulge_sfh = SyntheticSFH(base.ageBase)
bulge_sfh.addExp(14e9, 2.0e9, 1.0)
bulge_flux = (base.f_ssp * bulge_sfh.massVector()[:, np.newaxis]).sum(axis=1).sum(axis=0)
bulge_flux /= np.median(bulge_flux[wl_norm_window])

disk_sfh = SyntheticSFH(base.ageBase)
disk_sfh.addExp(10e9, 2.0e9, 1.0)
disk_flux = (base.f_ssp * disk_sfh.massVector()[:, np.newaxis]).sum(axis=1).sum(axis=0)
disk_flux /= np.median(disk_flux[wl_norm_window])

logger.debug('Plotting SFH.')
fig = plt.figure(figsize=(8,6))
gs = plt.GridSpec(2, 1, height_ratios=[1.0, 1.0])
ax = plt.subplot(gs[0])
age_Gyr = base.ageBase / 1e9
ax.plot(age_Gyr, bulge_sfh.massVector(), 'r-', label='bulge')
ax.plot(age_Gyr, disk_sfh.massVector(), 'b-', label='disk')
ax.set_xlim(age_Gyr.min(), age_Gyr.max())
ax.set_xlabel(r'Age [Gyr]')
ax.set_ylabel(r'SFR (weight)')
ax.set_title(r'Star formation history')
ax.legend(loc='upper left')

ax = plt.subplot(gs[1])
ax.plot(base.l_ssp, bulge_flux, 'r-')
ax.plot(base.l_ssp, disk_flux, 'b-')
ax.set_xlabel(r'Wavelength [$\AA$]')
ax.set_ylabel(r'Relative flux')
ax.set_title(r'Spectra')
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig()

################################################################################
##########
########## Morphology model setup
##########
################################################################################

logger.info('Creating true B-D model.')
flux_unit = 1e-16
shape = (72,77)
x0 = 36.0
y0 = 33.0
I_e = 2.0
r_e = 5.0
n = 2.0
I_0 = 1.0
h = 15.0
ba = 0.9
ell = 1.0 - ba
pa = 45.0
pa_rad = pa / 180 * np.pi
flagged = distance(shape, x0, y0) > 32.0
noise = 0.05



true_model = BDModel(0, x0, y0,
                     I_e=I_e, r_e=r_e, n=n, PA_b=pa, ell_b=ell,
                     I_0=I_0, h=h, PA_d=pa, ell_d=ell)
logger.info('True model (wavelength independent):\n%s\n' % str(true_model))

logger.info('Creating model images.')
PSF = gaussian_psf(2.4, size=9)
bulge_image, disk_image = create_model_images(true_model, shape, PSF, flux_unit=1.0)
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

logger.debug('Plotting true model.')
vmin = np.log10(model_image.min())
vmax = np.log10(model_image.max())
fig = plt.figure(figsize=(8, 6))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
ax = plt.subplot(gs[0,0])
ax.imshow(np.log10(model_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Total')

ax = plt.subplot(gs[0,1])
ax.imshow(np.log10(bulge_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Bulge')

ax = plt.subplot(gs[0,2])
ax.imshow(np.log10(disk_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Disk')

ax = plt.subplot(gs[1,:])
bins = np.arange(0, 32)
bins_c = bins[:-1] + 0.5
mr = radialProfile(np.log10(model_image), bins, x0, y0, pa, ba)
br = radialProfile(np.log10(bulge_image), bins, x0, y0, pa, ba)
dr = radialProfile(np.log10(disk_image), bins, x0, y0, pa, ba)
ax.plot(bins_c, mr, 'k-', label='Total')
ax.plot(bins_c, br, 'r-', label='Bulge')
ax.plot(bins_c, dr, 'b-', label='Disk')
ax.set_xlabel(r'Radius [arcsec]')
ax.set_ylabel(r'$\log$ flux (relative)')
ax.set_xlim(0.0, 30.0)
ax.set_ylim(-1.0, 1.0)
ax.legend(loc='upper right')

plt.suptitle(r'True model: $I_e = %.3f$, $r_e = %.3f$, $n = %.3f$, $I_0 = %.3f$, $h = %.3f$' % (I_e, r_e, n, I_0, h))
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig()

################################################################################
##########
########## Decomposition 
##########
################################################################################

logger.info('Beginning decomposition.')
decomp = IFSDecomposer()
decomp.setSynthPSF(FWHM=2.4, size=9)
decomp.loadData(base.l_ssp, full_spectra, full_noise, np.zeros_like(full_spectra, dtype='bool'))

swll, swlu = 5590.0, 5680.0
sl1 = find_nearest_index(decomp.wl, swll)
sl2 = find_nearest_index(decomp.wl, swlu)
qSignal, qNoise, qWl = decomp.getSpectraSlice(sl1, sl2)

logger.warn('Computing initial model (takes a LOT of time).')
t1 = time.time()
initial_model = bd_initial_model(qSignal, qNoise, PSF, x0, y0)
bulge_image, disk_image = create_model_images(initial_model, qSignal.shape, PSF, flux_unit=1.0)
logger.info('Refined initial model:\n%s\n' % initial_model)
logger.warn('Initial model time: %.2f\n' % (time.time() - t1))

logger.debug('Plotting guessed initial model.')
vmin = np.log10(qSignal.min())
vmax = np.log10(qSignal.max())
fig = plt.figure(figsize=(8, 6))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
ax = plt.subplot(gs[0,0])
ax.imshow(np.log10(model_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Total')

ax = plt.subplot(gs[0,1])
ax.imshow(np.log10(bulge_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Bulge')

ax = plt.subplot(gs[0,2])
ax.imshow(np.log10(disk_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Disk')

ax = plt.subplot(gs[1,:])
bins = np.arange(0, 32)
bins_c = bins[:-1] + 0.5
pa_i, ba_i = ellipse_params(qSignal, x0, y0)
mr = radialProfile(np.log10(qSignal), bins, x0, y0, pa_i, ba_i)
br = radialProfile(np.log10(bulge_image), bins, x0, y0, pa_i, ba_i)
dr = radialProfile(np.log10(disk_image), bins, x0, y0, pa_i, ba_i)
ax.plot(bins_c, mr, 'k-', label='Total')
ax.plot(bins_c, br, 'r-', label='Bulge')
ax.plot(bins_c, dr, 'b-', label='Disk')
ax.set_xlabel(r'Radius [arcsec]')
ax.set_ylabel(r'$\log$ flux (relative)')
ax.set_xlim(0.0, 30.0)
ax.set_ylim(-1.0, 1.0)
ax.legend(loc='upper right')

tmp = (initial_model.bulge.I_e.value,
       initial_model.bulge.r_e.value,
       initial_model.bulge.n.value,
       initial_model.disk.I_0.value, initial_model.disk.h.value)
plt.suptitle(r'Initial model: $I_e = %.3f$, $r_e = %.3f$, $n = %.3f$, $I_0 = %.3f$, $h = %.3f$' % tmp)
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig()

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

logger.info('Computing model spectra.')
fitted_bulge_spectra, fitted_disk_spectra = decomp.getModelSpectra(models)


################################################################################
##########
########## Decomposition plots 
##########
################################################################################

logger.info('Plotting stuff.')

colnames = [
            'I_0',
            'I_e',
            'n',
            'h',
            'r_e',
            'x0',
            'PA_d',
            'PA_b',
            'y0',
            'ell_d',
            'ell_b',
            'chi2',
            ]

limits = {'I_e': (-17, -15),
          'r_e': (0, 20),
          'PA_b': (0, 180),
          'ell_b': (0.5, 1.0),
          'I_0': (-17, -15),
          'h': (0, 20),
          'PA_d': (0, 180),
          'ell_d': (0.5, 1.0),
          'x0': None,
          'y0': None,
          'chi2': None,
          'n_pix': None,
          'n': (1,5),
          }

ylabel = {'I_e': r'$\log I_e\ [erg / s /cm^2 / \AA]$',
          'r_e': r'$r_e\ [arcsec]$',
          'PA_b': r'$P.A.\ [degrees]$ (bulge)',
          'ell_b': r'$b/a$ (bulge)',
          'I_0': r'$\log I_0\ [erg / s /cm^2 / \AA]$',
          'h': r'$h\ [arcsec]$',
          'PA_d': r'$P.A.\ [degrees]$ (disk)',
          'ell_d': r'$b/a$ (disk)',
          'x0': r'$X_{center}\ [pixel]$',
          'y0': r'$Y_{center}\ [pixel]$',
          'chi2': r'$\chi^2$',
          'n_pix': r'$N_{pix}$',
          'n': r'Sersic index $(n)$',
          }

nothing = lambda x: x
rad_to_degrees = lambda x: x * 180.0 / np.pi
ell_to_ba = lambda x: 1 - x
log10flux = lambda x: np.log10(x*flux_unit)
func = {'I_e': log10flux,
        'r_e': nothing,
        'PA_b': nothing,
        'ell_b': ell_to_ba,
        'I_0': log10flux,
        'h': nothing,
        'PA_d': nothing,
        'ell_d': ell_to_ba,
        'x0': nothing,
        'y0': nothing,
        'chi2': nothing,
        'n_pix': nothing,
        'n': nothing,
        }

################################################################################
##########
########## All fit parameters 
##########
################################################################################
fig = plt.figure(figsize=(8, 7))
n_rows = 4
n_cols = 3
gs = plt.GridSpec(n_rows, n_cols)
true_params = np.array(true_model.getParams(), dtype=true_model.dtype)
for i, colname in enumerate(colnames):
    if colname is None: continue
    ax = plt.subplot(gs[i])
    y = func[colname](fitted_params[colname])
    y_1p = func[colname](first_pass_params[colname])
    ax.hlines(func[colname](true_params[colname]), decomp.wl.min(), decomp.wl.max(), linestyles=':', colors='k')
    ax.plot(first_pass_lambdas, y_1p, '.r')
    ax.plot(decomp.wl, y, 'k')
    ax.set_ylabel(ylabel[colname])
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    if (i / n_cols) == (n_rows - 1):
        ax.set_xlabel(r'wavelength $[\AA]$')
    else:
        ax.set_xticklabels([])
    if limits[colname] is not None:
        ymin = limits[colname][0]
        ymax = limits[colname][1]
    else:
        ymin = y.min()
        ymax = y.max()
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(decomp.wl.min(), decomp.wl.max())
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig(fig)

################################################################################
##########
########## Model images
##########
################################################################################

fig = plt.figure(figsize=(8, 6))
gs = plt.GridSpec(3, 2, height_ratios=[-0.2, 1.0, 1.0])

l_range = np.where((decomp.wl > 5590.0) & (decomp.wl < 5680.0))[0]
l1 = l_range[0]
l2 = l_range[-1]

bulge_im = np.median(fitted_bulge_spectra[l1:l2], axis=0)

disk_im = np.median(fitted_disk_spectra[l1:l2], axis=0)

total_im = np.median(decomp.flux[l1:l2], axis=0)

residual_im = (total_im - disk_im - bulge_im)  / total_im


def getMinMax(image):
    vals = np.ma.masked_invalid(image).compressed()
    mean = vals.mean()
    sigma = np.sqrt(vals.var())
    return mean - 3 * sigma, mean + 3 * sigma


vmin, vmax = getMinMax(np.log10(total_im))
res_vmin, res_vmax = getMinMax(residual_im)

ax = plt.subplot(gs[1,0])
im = ax.imshow(np.log10(bulge_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'$\log F^{bulge}_\lambda$')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[1,1])
im = ax.imshow(np.log10(disk_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'$\log F^{disk}_\lambda$')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,0])
im = ax.imshow(np.log10(total_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'$\log F^{obs}_\lambda$')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,1])
im = ax.imshow(residual_im, origin='lower', interpolation='nearest', cmap='RdBu', vmin=res_vmin, vmax=res_vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('$(F^{obs}_\lambda - F^{bulge}_\lambda - F^{disk}_\lambda) / F^{obs}_\lambda$')
plt.colorbar(im, ax=ax)

gs.tight_layout(fig, rect=[0, 0, 1, 0.9])
pdf.savefig(fig)

################################################################################
##########
########## Fit quality
##########
################################################################################

fig = plt.figure(figsize=(8, 10))
gs = plt.GridSpec(4, 1, height_ratios=[-0.2, 1.0, 1.0, 1.0])

ax = plt.subplot(gs[1])
f_total = decomp.flux[:,37,37]
f_disk = fitted_disk_spectra[:,37,37]
f_disk_orig = disk_spectra[:,37,37]
f_bulge = fitted_bulge_spectra[:,37,37]
f_bulge_orig = bulge_spectra[:,37,37]
f_res = f_total - f_disk - f_bulge
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(decomp.wl, f_total, 'k', label='observed')
ax.plot(decomp.wl, f_disk, 'b', label='disk model')
ax.plot(decomp.wl, f_disk_orig, 'b--', label='original disk')
ax.plot(decomp.wl, f_bulge, 'r', label='bulge model')
ax.plot(decomp.wl, f_bulge_orig, 'r--', label='original bulge')
ax.plot(decomp.wl, f_res, 'm', label='residual')
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
ax.legend()

ax = plt.subplot(gs[2])
I_e = func['I_e'](fitted_params['I_e'])
I_0 = func['I_0'](fitted_params['I_0'])
vmin = min(I_e.min(), I_0.min())
vmax = max(I_e.max(), I_0.max())
ax.plot(decomp.wl, I_0, 'b', label=r'Disk $(I_{0})$')
ax.plot(decomp.wl, I_e, 'r', label=r'Bulge $(I_{e})$')
if limits['I_0'] is not None:
    ymin = limits['I_0'][0]
    ymax = limits['I_0'][1]
else:
    ymin = y.min()
    ymax = y.max()
ax.set_ylim(ymin, ymax)
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$\log\ I$')
ax.legend()

ax = plt.subplot(gs[3])
r_e = func['r_e'](fitted_params['r_e'])
h = func['h'](fitted_params['h'])
vmin = min(r_e.min(), h.min())
vmax = max(r_e.max(), h.max())
ax.plot(decomp.wl, h, 'b', label=r'Disk $(h)$')
ax.plot(decomp.wl, r_e, 'r', label=r'Bulge $(r_{e})$')
if limits['r_e'] is not None:
    ymin = limits['r_e'][0]
    ymax = limits['r_e'][1]
else:
    ymin = y.min()
    ymax = y.max()
ax.set_ylim(ymin, ymax)
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xlabel(r'wavelength $[\AA]$')
ax.set_ylabel(r'radius $[arcsec]$')
ax.legend()
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig(fig)

################################################################################
##########
########## Radial profiles
##########
################################################################################

fig = plt.figure(figsize=(8, 10))
N_cols = 5
N_rows = 6
N_cell = N_cols * N_rows
delta_l = decomp.Nl_obs / N_cell
l_bins = np.arange(0, decomp.Nl_obs, delta_l)
gs = plt.GridSpec(N_rows, N_cols)
bin_r = np.arange(30)
bin_c = bin_r[:-1] + 0.5

for i in xrange(N_cols):
    for j in xrange(N_rows):
        ax = plt.subplot(gs[j, i])
        cell = j * N_cols + i
        l1 = l_bins[cell]
        if cell < N_cell:
            l2 = l_bins[cell+1]
        else:
            l2 = decomp.Nl_obs - 1
        l = l1
        while l < l2:
            if not fitted_params['flag'][l]:
                break
            l += 1
        else:
            print 'Only flagged stuff in the interval [%d:%d]' % (decomp.wl[l1], decomp.wl[l2])
            l = l1
        wl = decomp.wl[l]
        x0 = fitted_params['x0'][l]
        y0 = fitted_params['y0'][l]
        bulge_im = fitted_bulge_spectra[l]
        disk_im = fitted_disk_spectra[l]
        total_im = decomp.flux[l]
        mask = ~np.isnan(total_im)
        pa, ba = ellipse_params(total_im, x0, y0)
        r__yx = distance(total_im.shape, x0, y0, pa, ba)
        bulge_r = radialProfile(bulge_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        disk_r = radialProfile(disk_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        total_r = radialProfile(total_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        ax.plot(bin_c, np.log10(total_r), 'k', label='observed')
        ax.plot(bin_c, np.log10(disk_r + bulge_r), 'k:', label='model')
        ax.plot(bin_c, np.log10(disk_r), 'b:', label='disk model')
        ax.plot(bin_c, np.log10(bulge_r), 'r:', label='bulge model')
        ax.text(0.5, 0.85, r'$%d\ \AA$' % wl, transform=ax.transAxes)
        ax.set_ylim(-17, -14.5)
        ax.set_xlim(0, bin_c.max())
        if i == 0 and j == (N_rows - 1):
            ax.set_xlabel(r'radius $[arcsec]$')
            ax.set_ylabel(r'$\log F_\lambda\ [erg / s / cm^2 / \AA]$')
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig(fig)

pdf.close()


