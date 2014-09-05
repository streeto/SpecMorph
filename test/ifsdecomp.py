'''
Created on 30/05/2014

@author: andre
'''


from specmorph.model import bd_initial_model, create_model_images, smooth_models, BDModel
from specmorph.decomposition import IFSDecomposer
from specmorph.geometry import distance, ellipse_params
from specmorph.util import logger, find_nearest_index

from tables import openFile
from pycasso.util import radialProfile
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import argparse
import sys

################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='Mock Bulge/Disk decomposition.')
    
    parser.add_argument('--db', dest='db', default='fake_ifs.h5',
                        help='Output HDF5 database path.')
    parser.add_argument('--param-degree', dest='paramDegree', type=int, default=1,
                        help='Degree of polynomial to fit morphology parameters.')
    parser.add_argument('--name', dest='galaxyName', default='default_galaxy',
                        help='Output HDF5 database path.')
    parser.add_argument('--plot', dest='plotFile', default='test.pdf',
                        help='Plot to this file.')
    parser.add_argument('--psf-fwhm', dest='modelPsfFWHM', type=float, default=2.4,
                        help='PSF FWHM (arcseconds) used when modeling.')
    parser.add_argument('--fix-structural', dest='fixStructural', action='store_true',
                        help='Fix structural parameters in second pass.')
    parser.add_argument('--cache', dest='cacheModel', default=None,
                        help='Use the specified file as cached initial model.')
    
    return parser.parse_args()
################################################################################


################################################################################
def plot_setup(plot_file):
    pdf = PdfPages(plot_file)
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
    return pdf
################################################################################


################################################################################
def line(x0, y0, a, x):
    '''
    Creates a line passing through ``(x0, y0)`` with
    angle ``tan(theta) = a``.
    
    Parameters
    ----------
    x0, y0 : float
        Coordinates of the root point of the line.
        
    a : float
        Tangent of the angle between the line and
        the X axis.
        
    x : array
        X-coordinates of the line to be created.
        
    Return
    ------
    y : array
        Y coordinates of the line, same shape as ``x``.
    '''
    return (x - x0) * a + y0
################################################################################


################################################################################
##########
########## Setup
##########
################################################################################

logger.setLevel(-1)
args = parse_args()
pdf = plot_setup(args.plotFile)

logger.info('Loading data from: %s' % args.db)
db = openFile(args.db)

try:
    logger.info('Galaxy: %s' % args.galaxyName)
    g = db.getNode('/%s' % args.galaxyName)
except:
    logger.error('Unknown galaxy: %s' % args.galaxyName)
    sys.exit()

bulge_image = np.ma.array(g.bulge_image[...])
disk_image = np.ma.array(g.disk_image[...])
r = distance(bulge_image.shape, bulge_image.shape[1] / 2, bulge_image.shape[0] / 2)
masked = r > 32.0
bulge_image[masked] = np.ma.masked
disk_image[masked] = np.ma.masked
full_image = bulge_image + disk_image

bulge_ifs = np.ma.array(g.bulge_ifs[...])
bulge_ifs[:,masked] = np.ma.masked
bulge_ifs_nopsf = np.ma.array(g.bulge_ifs_nopsf[...])
bulge_ifs_nopsf[:,masked] = np.ma.masked

disk_ifs = np.ma.array(g.disk_ifs[...])
disk_ifs[:,masked] = np.ma.masked
disk_ifs_nopsf = np.ma.array(g.disk_ifs_nopsf[...])
disk_ifs_nopsf[:,masked] = np.ma.masked

full_ifs = np.ma.array(g.full_ifs[...])
full_ifs[:,masked] = np.ma.masked
full_ifs_noise = np.ma.array(g.full_ifs_noise[...])
full_ifs_noise[:,masked] = np.ma.masked

wl = g.wl[...]
true_psf = np.ma.array(g.psf[...])
tau_image = g.tau_image[...]
age_base = g.age_base[...]

flux_unit = g.full_ifs.attrs.fluxUnit
true_psf_FWHM = g.full_ifs.attrs.psfFWHM
norm_params = g.full_ifs.attrs.model
norm_model = BDModel.fromParamVector(norm_params)
norm_x0 = norm_params['x0']
norm_y0 = norm_params['y0']

db.close()

logger.debug('Plotting original model.')
#index_norm = find_nearest_index(l_ssp, 5635.0)
vmin = np.log10(full_image.min())
vmax = np.log10(full_image.max())
fig = plt.figure(figsize=(8, 6))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
ax = plt.subplot(gs[0,0])
ax.imshow(np.log10(full_image), vmin=vmin, vmax=vmax)
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

ax = plt.subplot(gs[1,:2])
bins = np.arange(0, 32)
bins_c = bins[:-1] + 0.5
pa, ell = ellipse_params(full_image, norm_x0, norm_y0)
pa = (90.0 + pa) * np.pi / 180.0
ba = 1.0 - ell
mr = radialProfile(np.log10(full_image), bins, norm_x0, norm_y0, pa, ba)
br = radialProfile(np.log10(bulge_image), bins, norm_x0, norm_y0, pa, ba)
dr = radialProfile(np.log10(disk_image), bins, norm_x0, norm_y0, pa, ba)
ax.plot(bins_c, mr, 'k-', label='Total')
ax.plot(bins_c, br, 'r-', label='Bulge')
ax.plot(bins_c, dr, 'b-', label='Disk')
ax.set_xlabel(r'Radius [arcsec]')
ax.set_ylabel(r'$\log$ flux (relative)')
ax.set_xlim(0.0, 30.0)
ax.set_ylim(-1.0, 1.1)
ax.legend(loc='upper right')

ax = plt.subplot(gs[1,2])
im = ax.imshow(tau_image)
plt.colorbar(im, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'$\tau$ (SFH bulge)')


norm_I_e = norm_model.bulge.I_e.value
norm_r_e = norm_model.bulge.r_e.value
norm_n = norm_model.bulge.n.value
norm_I_0 = norm_model.disk.I_0.value
norm_h = norm_model.disk.h.value
tmp = (norm_I_e, norm_r_e, norm_n, norm_I_0, norm_h, true_psf_FWHM)
plt.suptitle(r'Original model: $I_e = %.3f$, $r_e = %.3f$, $n = %.3f$, $I_0 = %.3f$, $h = %.3f$, $FWHM = %.2f$' % tmp)
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig()

################################################################################
##########
########## Decomposition 
##########
################################################################################

logger.info('Beginning decomposition.')
decomp = IFSDecomposer()
logger.info('Model using PSF FWHM = %.2f ".' % args.modelPsfFWHM)
decomp.setSynthPSF(FWHM=args.modelPsfFWHM, size=9)
decomp.loadData(wl, full_ifs, full_ifs_noise, np.zeros_like(full_ifs, dtype='bool'))

swll, swlu = 5590.0, 5680.0
sl1 = find_nearest_index(decomp.wl, swll)
sl2 = find_nearest_index(decomp.wl, swlu)
qSignal, qNoise, qWl = decomp.getSpectraSlice(sl1, sl2)

logger.warn('Computing initial model (takes a LOT of time).')
t1 = time.time()
initial_model = bd_initial_model(qSignal, qNoise, decomp.PSF, quiet=False, cache_model_file=args.cacheModel)
bulge_image, disk_image = create_model_images(initial_model, qSignal.shape, decomp.PSF)
logger.warn('Initial model time: %.2f\n' % (time.time() - t1))

logger.debug('Plotting guessed initial model.')
vmin = np.log10(qSignal.min())
vmax = np.log10(qSignal.max())
fig = plt.figure(figsize=(8, 6))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
ax = plt.subplot(gs[0,0])
ax.imshow(np.log10(qSignal), vmin=vmin, vmax=vmax)
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
y0 = initial_model.y0.value
x0 = initial_model.x0.value
pa_i, ell_i = ellipse_params(qSignal, x0, y0)
pa_i = (90.0 + pa_i) * np.pi / 180.0
ba_i = 1.0 - ell_i
mr = radialProfile(np.log10(qSignal), bins, x0, y0, pa_i, ba_i)
br = radialProfile(np.log10(bulge_image), bins, x0, y0, pa_i, ba_i)
dr = radialProfile(np.log10(disk_image), bins, x0, y0, pa_i, ba_i)
ax.plot(bins_c, mr, 'k-', label='Total')
ax.plot(bins_c, br, 'r-', label='Bulge')
ax.plot(bins_c, dr, 'b-', label='Disk')
ax.set_xlabel(r'Radius [arcsec]')
ax.set_ylabel(r'$\log$ flux (relative)')
ax.set_xlim(0.0, 30.0)
ax.set_ylim(-1.0, 1.1)
ax.legend(loc='upper right')

tmp = (initial_model.bulge.I_e.value,
       initial_model.bulge.r_e.value,
       initial_model.bulge.n.value,
       initial_model.disk.I_0.value,
       initial_model.disk.h.value,
       args.modelPsfFWHM)
plt.suptitle(r'Initial model: $I_e = %.3f$, $r_e = %.3f$, $n = %.3f$, $I_0 = %.3f$, $h = %.3f$, $FWHM = %.2f$' % tmp)
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig()

logger.info('Starting first pass modeling.')
t1 = time.time()
first_pass_models = decomp.fitSpectra(step=100, box_radius=50, initial_model=initial_model, mode='NM')
first_pass_params = np.array([m.getParams() for m in first_pass_models], dtype=first_pass_models[0].dtype)
first_pass_lambdas = decomp.wl[::100]
logger.info('Done first pass modeling, time: %.2f' % (time.time() - t1))

logger.info('Smoothing parameters with polynomial of degree %d.' % args.paramDegree)
smoothed_models = smooth_models(first_pass_models, decomp.wl,
                                degree=args.paramDegree, fix_structural=args.fixStructural)
smoothed_params = np.array([m.getParams() for m in smoothed_models], dtype=smoothed_models[0].dtype)
        
logger.info('Starting second pass modeling...')
t1 = time.time()
fitted_models = decomp.fitSpectra(step=1, box_radius=0, initial_model=smoothed_models, mode='LM', insist=True)
fitted_params = np.array([m.getParams() for m in fitted_models], dtype=fitted_models[0].dtype)
logger.info('Done second pass modeling, time: %.2f' % (time.time() - t1))

logger.info('Computing model spectra.')
fitted_bulge_ifs, fitted_disk_ifs = decomp.getModelSpectra(fitted_models)
fitted_bulge_ifs_nopsf, fitted_disk_ifs_nopsf = decomp.getModelSpectra(fitted_models, use_PSF=False)

logger.info('Average fit results:')
print_params = ('I_e', 'r_e', 'n', 'PA_b', 'ell_b', 'I_0', 'h', 'PA_d', 'ell_d', )
for p in fitted_params.dtype.names:
    if p not in print_params: continue
    logger.info('    %s = %.3f +/- %.3f' % (p, np.mean(fitted_params[p]), np.std(fitted_params[p])))


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
for i, colname in enumerate(colnames):
    if colname is None: continue
    ax = plt.subplot(gs[i])
    y_orig = np.ones_like(decomp.wl) * func[colname](norm_params[colname])
    y = func[colname](fitted_params[colname])
    y_1p = func[colname](first_pass_params[colname])
    ax.plot(decomp.wl, y_orig, ':k')
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

bulge_im = np.median(fitted_bulge_ifs[l1:l2], axis=0)

disk_im = np.median(fitted_disk_ifs[l1:l2], axis=0)

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
gs = plt.GridSpec(3, 2, height_ratios=[1.0, 1.0, 1.0])

ax = plt.subplot(gs[0, :])
xx = np.round(initial_model.x0.value)
yy = np.round(initial_model.y0.value)
f_total = decomp.flux[:,yy,xx]
f_disk = fitted_disk_ifs[:,yy,xx]
f_disk_orig = disk_ifs[:,yy,xx]
f_bulge = fitted_bulge_ifs[:,yy,xx]
f_bulge_orig = bulge_ifs[:,yy,xx]
f_res = f_total - f_disk - f_bulge
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(decomp.wl, f_total, 'k', label='observed')
ax.plot(decomp.wl, f_res, 'm', label='residual')
ax.plot(decomp.wl, f_disk, 'b', label='disk model')
ax.plot(decomp.wl, f_disk_orig, 'b:', label='original disk')
ax.plot(decomp.wl, f_bulge, 'r', label='bulge model')
ax.plot(decomp.wl, f_bulge_orig, 'r:', label='original bulge')
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
ax.set_title(r'Spectra at the nucleus')
ax.legend()

ax = plt.subplot(gs[1, :])
xx = np.ceil(initial_model.x0.value + initial_model.bulge.r_e.value)
f_total = decomp.flux[:,yy,xx]
f_disk = fitted_disk_ifs[:,yy,xx]
f_disk_orig = disk_ifs[:,yy,xx]
f_bulge = fitted_bulge_ifs[:,yy,xx]
f_bulge_orig = bulge_ifs[:,yy,xx]
f_res = f_total - f_disk - f_bulge
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(decomp.wl, f_total, 'k', label='observed')
ax.plot(decomp.wl, f_res, 'm', label='residual')
ax.plot(decomp.wl, f_disk, 'b', label='disk model')
ax.plot(decomp.wl, f_disk_orig, 'b:', label='original disk')
ax.plot(decomp.wl, f_bulge, 'r', label='bulge model')
ax.plot(decomp.wl, f_bulge_orig, 'r:', label='original bulge')
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xlabel(r'wavelength $[\AA]$')
ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
ax.set_title(r'Spectra at $R = r_e$ ($%.1f\,arcsec$)' % initial_model.bulge.r_e.value)
ax.legend()

ax = plt.subplot(gs[2, 0])
bulge_error = radialProfile(fitted_bulge_ifs/bulge_ifs, bins, norm_x0, norm_y0, pa, ba, rad_scale=1.0)
im = ax.pcolormesh(bins, wl, bulge_error)
ax.set_xlim(0, bins.max())
ax.set_ylim(wl.min(), wl.max())
ax.set_xlabel(r'radius $[arcsec]$')
ax.set_ylabel(r'wavelength $[\AA]$')
ax.set_title(r'Fitted / original ratio for bulge')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2, 1])
disk_error = radialProfile(fitted_disk_ifs/disk_ifs, bins, norm_x0, norm_y0, pa, ba, rad_scale=1.0)
im = ax.pcolormesh(bins, wl, disk_error)
ax.set_xlim(0, bins.max())
ax.set_ylim(wl.min(), wl.max())
ax.set_xlabel(r'radius $[arcsec]$')
ax.set_ylabel(r'wavelength $[\AA]$')
ax.set_title(r'Fitted / original ratio for disk')
plt.colorbar(im, ax=ax)

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
        plot_wl = decomp.wl[l]
        x0 = fitted_params['x0'][l]
        y0 = fitted_params['y0'][l]
        bulge_im = fitted_bulge_ifs[l]
        disk_im = fitted_disk_ifs[l]
        total_im = decomp.flux[l]
        mask = ~np.isnan(total_im)
        pa, ell = ellipse_params(total_im, x0, y0)
        pa = (90.0 + pa) * np.pi / 180.0
        ba = 1.0 - ell
        r__yx = distance(total_im.shape, x0, y0, pa, ba)
        bulge_r = radialProfile(bulge_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        disk_r = radialProfile(disk_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        total_r = radialProfile(total_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        ax.plot(bin_c, np.log10(total_r), 'k', label='observed')
        ax.plot(bin_c, np.log10(disk_r + bulge_r), 'k:', label='model')
        ax.plot(bin_c, np.log10(disk_r), 'b:', label='disk model')
        ax.plot(bin_c, np.log10(bulge_r), 'r:', label='bulge model')
        ax.text(0.5, 0.85, r'$%d\ \AA$' % plot_wl, transform=ax.transAxes)
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


