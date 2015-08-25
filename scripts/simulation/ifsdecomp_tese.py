# -*- coding: utf-8 -*-
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
from matplotlib.ticker import MultipleLocator
import argparse
import sys
from os import path
from pystarlight.util.base import StarlightBase
from specmorph.components import SyntheticSFH
from pystarlight.util.StarlightUtils import bin_edges, hist_resample

width_pt = 448.07378
width_in = width_pt / 72.0 * 0.95

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
    parser.add_argument('--psf-fwhm', dest='modelPsfFWHM', type=float, default=2.9,
                        help='PSF FWHM (arcseconds) used when modeling.')
    parser.add_argument('--psf-beta', dest='modelPsfBeta', type=float, default=4.0,
                        help='PSF beta used when modeling.')
    parser.add_argument('--cache', dest='cacheModel', default=None,
                        help='Use the specified file as cached initial model.')
    parser.add_argument('--base', dest='baseFile', default='data/starlight/BASE.gsd6e',
                        help='File describing the starlight bases.')
    parser.add_argument('--base-dir', dest='baseDir', default='data/starlight/BasesDir',
                        help='Directory containing the base spectra.')
    parser.add_argument('--tau0', dest='tau0', type=float, default=2e9,
                        help='Width of the bulge star formation burst at the nucleus.')
    parser.add_argument('--dtau-dr', dest='dtau_dr', type=float, default=2e8,
                        help='Gradient of tau.')
    
    return parser.parse_args()
################################################################################


################################################################################
def tau_r(tau0, dtau_dr, r):
    return tau0 + r * dtau_dr
################################################################################


################################################################################
def get_synth_sfh(t0, tau, ages, dt=0.5e9):
    sfh = SyntheticSFH(ages)
    sfh.addExp(t0, tau, 1.0)
    return sfh.massVector()
################################################################################


################################################################################
def smooth_sfh(mu, ages, dt=0.5e9):
    logtc_bins = bin_edges(np.log10(ages))
    tc_bins = 10**logtc_bins
    tl = np.arange(tc_bins.min(), tc_bins.max()+dt, dt)
    tl_bins = bin_edges(tl)
    mu_res = hist_resample(tc_bins, tl_bins, mu)
    SFR = mu_res / dt
    # Add bondary points so that np.trapz(SFR, tl) == Mini.sum().
    SFR = np.hstack((0, SFR, 0))
    tl = np.hstack((tl[0] - dt, tl, tl[-1] + dt))
    return SFR, tl    
################################################################################


################################################################################
def plot_setup():
    plotpars = {'legend.fontsize': 10,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'text.fontsize': 10,
                'axes.titlesize': 12,
                'lines.linewidth': 0.5,
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
plot_setup()

logger.info('Loading data from: %s' % args.db)
db = openFile(args.db)

try:
    logger.info('Galaxy: %s' % args.galaxyName)
    g = db.getNode('/%s' % args.galaxyName)
except:
    logger.error('Unknown galaxy: %s' % args.galaxyName)
    sys.exit()

masked = (g.flag_ifs[...] > 0) | (g.full_ifs[...] < 0.0)
masked2d = masked.sum(axis=0) > masked.shape[0] / 2
bulge_image = g.bulge_image[...]
disk_image = g.disk_image[...]
full_image = bulge_image + disk_image

bulge_ifs = g.bulge_ifs[...]
bulge_ifs_nopsf = g.bulge_ifs_nopsf[...]

disk_ifs = g.disk_ifs[...]
disk_ifs_nopsf = g.disk_ifs_nopsf[...]

full_ifs = np.ma.array(g.full_ifs[...], mask=masked)
full_ifs_noise = np.ma.array(g.full_ifs_noise[...], mask=masked)

wl = g.wl[...]
true_psf = np.ma.array(g.psf[...])
tau_image = g.tau_image[...]
age_base = g.age_base[...]

flux_unit = g.full_ifs.attrs.fluxUnit
true_psf_FWHM = g.full_ifs.attrs.psfFWHM
norm_params = g.full_ifs.attrs.model
norm_model = BDModel.fromParamVector(norm_params)
norm_x0 = norm_params['x0'] - 1
norm_y0 = norm_params['y0'] - 1

db.close()

################################################################################
##########
########## Plot original morph. model 
##########
################################################################################
logger.debug('Plotting original model.')
#index_norm = find_nearest_index(l_ssp, 5635.0)
vmin = np.log10(full_image.min())
vmax = np.log10(full_image.max())
fig = plt.figure(figsize=(width_in, 0.8 * width_in))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])

ax = plt.subplot(gs[0,0])
ax.imshow(np.log10(bulge_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Bojo')

ax = plt.subplot(gs[0,1])
ax.imshow(np.log10(disk_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Disco')

ax = plt.subplot(gs[0,2])
ax.imshow(np.log10(full_image), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Total')

ax = plt.subplot(gs[1,:])
bins = np.arange(0, 32)
bins_c = bins[:-1]
pa, ell = ellipse_params(full_image, norm_x0, norm_y0)
pa = (90.0 + pa) * np.pi / 180.0
ba = 1.0 - ell
mr = radialProfile(np.log10(full_image), bins, norm_x0, norm_y0, pa, ba)
br = radialProfile(np.log10(bulge_image), bins, norm_x0, norm_y0, pa, ba)
dr = radialProfile(np.log10(disk_image), bins, norm_x0, norm_y0, pa, ba)
ax.plot(bins_c, mr, 'k-', label='Total')
ax.plot(bins_c, br, 'r-', label='Bojo')
ax.plot(bins_c, dr, 'b-', label='Disco')
ax.set_xlabel(r'Raio $[\mathrm{arcsec}]$')
ax.set_ylabel(r'$\log$ Fluxo')
ax.set_xlim(0.0, 20.0)
ax.set_ylim(-1.0, 1.99)
ax.legend(loc='upper right', frameon=False)

norm_I_e = norm_model.bulge.I_e.value
norm_r_e = norm_model.bulge.r_e.value
norm_n = norm_model.bulge.n.value
norm_I_0 = norm_model.disk.I_0.value
norm_h = norm_model.disk.h.value
plt.suptitle(r'Modelo original (sem convoluir com PSF)')
gs.tight_layout(fig, rect=[0, 0, 1, 0.95])
plt.savefig('plots/tese/simulation_initmodel.pdf')

################################################################################
##########
########## Plot original population model 
##########
################################################################################
logger.debug('Plotting original population model.')
#index_norm = find_nearest_index(l_ssp, 5635.0)

logger.info('Loading base %s', path.basename(args.baseFile))
t1 = time.clock()
base = StarlightBase(args.baseFile, args.baseDir)
wl = np.arange(3650.0, 6850.0, 2.0)
f_ssp = base.f_sspResam(wl)
logger.info('Took %.2f seconds to read the base (%d files)' % (time.clock() - t1, base.sspfile.size))
wl_norm_window = (wl < 5680.0) & (wl > 5590.0)

t0 = base.ageBase.max()

r = np.arange(30)
tau = tau_r(args.tau0, args.dtau_dr, r)
print tau

bulge_sfh_tau0 = get_synth_sfh(t0, args.tau0, base.ageBase)
bulge_sfh_tau1 = get_synth_sfh(t0, 4e9, base.ageBase)
bulge_sfh_tau2 = get_synth_sfh(t0, 8e9, base.ageBase)
disk_sfh_tauInf = get_synth_sfh(t0, 1e12, base.ageBase)

bulge_flux_tau0 = (f_ssp * bulge_sfh_tau0[:, np.newaxis]).sum(axis=1).sum(axis=0)
bulge_flux_tau0 /= bulge_flux_tau0.max()
bulge_flux_tau1 = (f_ssp * bulge_sfh_tau2[:, np.newaxis]).sum(axis=1).sum(axis=0)
bulge_flux_tau1 /= bulge_flux_tau1.max()
bulge_flux_tau2 = (f_ssp * bulge_sfh_tau2[:, np.newaxis]).sum(axis=1).sum(axis=0)
bulge_flux_tau2 /= bulge_flux_tau2.max()
disk_flux_tauInf = (f_ssp * disk_sfh_tauInf[:, np.newaxis]).sum(axis=1).sum(axis=0)
disk_flux_tauInf /= disk_flux_tauInf.max()

fig = plt.figure(figsize=(width_in, 1.3 * width_in))
gs = plt.GridSpec(3, 1, height_ratios=[2.0, 2.0, 3.0])

ax = plt.subplot(gs[0])
ax.plot(r, tau / 1e9, 'k-')
ax.set_xlabel(r'Raio $[\mathrm{arcsec}]$')
ax.set_ylabel(r'$\tau\ [\mathrm{Ga}]$')
ax.set_xlim(0.0, r.max())
ax.set_ylim(0.0, 10)
ax.set_title(u'Escala de tempo de formação estelar (bojo)')
ax.legend(loc='lower right')
#plt.suptitle(r'Modelo original (sem convoluir com PSF)')

ax = plt.subplot(gs[1])

psi, t = smooth_sfh(bulge_sfh_tau0, base.ageBase)
ax.plot(t / 1e9, psi * 1e9, 'r-', label=r'$\tau = 2\,\mathrm{Ga}$ (bojo em $r = 0^{\prime\prime}$)')

psi, t = smooth_sfh(bulge_sfh_tau1, base.ageBase)
ax.plot(t / 1e9, psi * 1e9, 'r--', label=r'$\tau = 4\,\mathrm{Ga}$ (bojo em $r = 10^{\prime\prime}$)')

psi, t = smooth_sfh(bulge_sfh_tau2, base.ageBase)
ax.plot(t / 1e9, psi * 1e9, 'r:', label=r'$\tau = 8\,\mathrm{Ga}$ (bojo em $r = 30^{\prime\prime}$)')

psi, t = smooth_sfh(disk_sfh_tauInf, base.ageBase)
ax.plot(t / 1e9, psi * 1e9, 'b-', label=r'$\tau = \infty$ (disco)')

ax.set_xlabel(r'Tempo evolutivo $[\mathrm{Ga}]$')
ax.set_ylabel(r'$\Psi\ [\mathrm{M}_\odot / \mathrm{Ga}]$')
ax.set_xlim(0.0, 15)
#ax.set_ylim(-1.0, 1.99)
ax.set_title(u'Histórico de formação estelar')
ax.legend(loc='upper left', frameon=False)
#plt.suptitle(r'Modelo original (sem convoluir com PSF)')

ax = plt.subplot(gs[2])
shift = 0.3
ax.plot(wl, bulge_flux_tau0 + 3 * shift , 'r-')
ax.plot(wl, bulge_flux_tau1 + 2 * shift, 'r--')
ax.plot(wl, bulge_flux_tau2 + 1 * shift, 'r:')
ax.plot(wl, disk_flux_tauInf + 0 * shift, 'b-')
ax.set_xlabel(r'Comprimento de onda $[\mathrm{\AA}]$')
ax.set_ylabel(u'Fluxo (unid. arbitrárias)')
ax.set_yticklabels([])
ax.set_xlim(wl.min(), wl.max())
ax.set_title(u'Espectros de populações estelares compostas')
#ax.set_ylim(0.0, 10)
#plt.suptitle(r'Modelo original (sem convoluir com PSF)')

gs.tight_layout(fig, rect=[0, 0, 1, 1.0])
plt.savefig('plots/tese/simulation_popmodel.pdf')

################################################################################
##########
########## Decomposition 
##########
################################################################################

logger.info('Beginning decomposition.')
decomp = IFSDecomposer()
logger.info('Model using PSF FWHM = %.2f ", beta = %.2f.' % (args.modelPsfFWHM, args.modelPsfBeta))
decomp.setSynthPSF(FWHM=args.modelPsfFWHM, beta=args.modelPsfBeta, size=15)
decomp.loadData(wl, full_ifs / flux_unit, full_ifs_noise / flux_unit, np.zeros_like(full_ifs, dtype='bool'))

swll, swlu = 5590.0, 5680.0
sl1 = find_nearest_index(decomp.wl, swll)
sl2 = find_nearest_index(decomp.wl, swlu)
qSignal, qNoise, qWl = decomp.getSpectraSlice(sl1, sl2)

logger.warn('Computing initial model (takes a LOT of time).')
t1 = time.time()
initial_model = bd_initial_model(qSignal, qNoise, decomp.PSF, quiet=False, cache_model_file=args.cacheModel)
bulge_image, disk_image = create_model_images(initial_model, qSignal.shape, PSF=decomp.PSF)
qSignal_model = bulge_image + disk_image
qSignal_residual = (qSignal - qSignal_model) / qSignal
bulge_image_nopsf, disk_image_nopsf = create_model_images(initial_model, qSignal.shape, PSF=None)
logger.warn('Initial model time: %.2f\n' % (time.time() - t1))

################################################################################
##########
########## Plot fitted morph. model 
##########
################################################################################
logger.debug('Plotting guessed initial model.')
vmin = np.log10(qSignal.min())
vmax = np.log10(qSignal.max())

fig = plt.figure(figsize=(width_in, 0.8 * width_in))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
ax = plt.subplot(gs[0,0])
ax.imshow(np.log10(qSignal), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Observado')

ax = plt.subplot(gs[0,1])
ax.imshow(np.log10(qSignal_model), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Modelo')

residual_range = np.abs(qSignal_residual).max()
res_vmin = - residual_range
res_vmax = residual_range

ax = plt.subplot(gs[0,2])
ax.imshow(qSignal_residual, vmin=res_vmin, vmax=res_vmax, cmap='RdBu')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(u'Resíduo')


ax = plt.subplot(gs[1,:])
bins = np.arange(0, 32)
bins_c = bins[:-1]
y0 = initial_model.y0.value - 1
x0 = initial_model.x0.value - 1
pa_i, ell_i = ellipse_params(qSignal, x0, y0)
pa_i = (90.0 + pa_i) * np.pi / 180.0
ba_i = 1.0 - ell_i

mr = radialProfile(np.log10(qSignal), bins, x0, y0, pa_i, ba_i)
br = radialProfile(np.log10(bulge_image_nopsf), bins, x0, y0, pa_i, ba_i)
dr = radialProfile(np.log10(disk_image_nopsf), bins, x0, y0, pa_i, ba_i)

ax.plot(bins_c, mr, 'k-', label='Observado')
ax.plot(bins_c, br, 'r-', label='Bojo')
ax.plot(bins_c, dr, 'b-', label='Disco')
ax.set_xlabel(r'Raio $[\mathrm{arcsec}]$')
ax.set_ylabel(r'$\log$ Fluxo')
ax.set_xlim(0.0, 20.0)
ax.set_ylim(-1.0, 1.99)
ax.legend(loc='upper right', frameon=False)

tmp = (initial_model.bulge.I_e.value,
       initial_model.bulge.r_e.value,
       initial_model.bulge.n.value,
       initial_model.disk.I_0.value,
       initial_model.disk.h.value,
       args.modelPsfFWHM)
#plt.suptitle(r'Initial model: $I_e = %.3f$, $r_e = %.3f$, $n = %.3f$, $I_0 = %.3f$, $h = %.3f$, $FWHM = %.2f$' % tmp)
plt.suptitle(r'Modelo ajustado em $5635\,\mathrm{\AA}$')
gs.tight_layout(fig, rect=[0, 0, 1, 0.95])

plt.savefig('plots/tese/simulation_fitmodel.pdf')


################################################################################
##########
########## Decomposition
##########
################################################################################
logger.info('Starting first pass modeling.')
t1 = time.time()
first_pass_models = decomp.fitSpectra(step=100, box_radius=50, initial_model=initial_model, mode='LM')
first_pass_params = np.array([m.getParams() for m in first_pass_models], dtype=first_pass_models[0].dtype)
first_pass_lambdas = decomp.wl[::100]
logger.info('Done first pass modeling, time: %.2f' % (time.time() - t1))

logger.info('Smoothing parameters with polynomial of degree %d.' % args.paramDegree)
smoothed_models = smooth_models(first_pass_models, decomp.wl, degree=args.paramDegree)
smoothed_params = np.array([m.getParams() for m in smoothed_models], dtype=smoothed_models[0].dtype)

logger.info('Starting second pass modeling...')
t1 = time.time()
fitted_models = decomp.fitSpectra(step=1, box_radius=0, initial_model=smoothed_models, mode='LM')
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
            'I_e',
            'I_0',
            'r_e',
            'h',
            'n',
            None,
            'PA_d',
            'PA_b',
            'ell_d',
            'ell_b',
            ]

limits = {'I_e': (-1, 1),
          'r_e': (0, 20),
          'PA_b': (40, 80),
          'ell_b': (0.0, 0.2),
          'I_0': (-1, 1),
          'h': (0, 20),
          'PA_d': (25, 65),
          'ell_d': (0.0, 0.2),
          'x0': None,
          'y0': None,
          'chi2': None,
          'n_pix': None,
          'n': (1,10),
          }

ylabel = {'I_e': r'$\log I_e$',
          'r_e': r'$r_e\ [\mathrm{arcsec}]$',
          'PA_b': r'P.A. $[\mathrm{graus}]$',
          'ell_b': r'$\epsilon$',
          'I_0': r'$\log I_0$',
          'h': r'$h\ [\mathrm{arcsec}]$',
          'PA_d': r'P.A. $[\mathrm{graus}]$',
          'ell_d': r'$\epsilon$',
          'x0': r'$X_0\ [pixel]$',
          'y0': r'$Y_0\ [pixel]$',
          'chi2': r'$\chi^2$',
          'n_pix': r'$N_{pix}$',
          'n': r'$n$',
          }

nothing = lambda x: x
rad_to_degrees = lambda x: x * 180.0 / np.pi
ell_to_ba = lambda x: 1 - x
log10flux = lambda x: np.log10(x)
func = {'I_e': log10flux,
        'r_e': nothing,
        'PA_b': nothing,
        'ell_b': nothing,
        'I_0': log10flux,
        'h': nothing,
        'PA_d': nothing,
        'ell_d': nothing,
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
fig = plt.figure(figsize=(width_in, 1.3 * width_in))
n_rows = 5
n_cols = 2
gs = plt.GridSpec(n_rows, n_cols)
for i, colname in enumerate(colnames):
    if colname is None: continue
    ax = plt.subplot(gs[i])
    y_orig = np.ones_like(decomp.wl) * func[colname](norm_params[colname])
    y = func[colname](fitted_params[colname])
    y_1p = func[colname](first_pass_params[colname])
    ax.plot(decomp.wl, y_orig, ':k')
    ax.plot(decomp.wl, y, 'k')
    ax.plot(first_pass_lambdas, y_1p, '.r')
    ax.set_ylabel(ylabel[colname])
    ax.xaxis.set_major_locator(MultipleLocator(1000))
    if (i / n_cols) == 0:
        if (i % n_cols) == 0:
            ax.set_title('Bojo')
        if (i % n_cols) == 1:
            ax.set_title('Disco')
    if (i / n_cols) == (n_rows - 1):
        ax.set_xlabel(r'Comprimento de onda $[\mathrm{\AA}]$')
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
plt.suptitle(u'Parâmetros morfológicos (PSF $FWHM = 2.6^{\prime\prime}$)')
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])

plt.savefig('plots/tese/simulation_fitparams.pdf')


################################################################################
##########
########## Spectra
##########
################################################################################
fig = plt.figure(figsize=(width_in, 1.25 * width_in))
gs = plt.GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.0])

ax = plt.subplot(gs[0])
xx = np.round(initial_model.x0.value)
yy = np.round(initial_model.y0.value)
f_total = decomp.flux[:,yy,xx] * flux_unit
f_disk = fitted_disk_ifs[:,yy,xx] * flux_unit
f_disk_orig = disk_ifs[:,yy,xx]
f_bulge = fitted_bulge_ifs[:,yy,xx] * flux_unit
f_bulge_orig = bulge_ifs[:,yy,xx]
f_res = f_total - f_disk - f_bulge
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(decomp.wl, f_total, 'k', label='observado')
ax.plot(decomp.wl, f_res, 'm', label=u'resíduo')
ax.plot(decomp.wl, f_disk, 'b', label='disco')
ax.plot(decomp.wl, f_disk_orig, 'b:', label=None)
ax.plot(decomp.wl, f_bulge, 'r', label='bojo')
ax.plot(decomp.wl, f_bulge_orig, 'r:', label=None)
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$F_\lambda\ [\mathrm{erg} / \mathrm{s} / \mathrm{cm}^2 / \mathrm{\AA}]$')
ax.set_title(r'Nuclear')
ax.legend(loc='center right', frameon=False)

ax = plt.subplot(gs[1])
xx = np.ceil(initial_model.x0.value + initial_model.bulge.r_e.value)
f_total = decomp.flux[:,yy,xx] * flux_unit
f_disk = fitted_disk_ifs[:,yy,xx] * flux_unit
f_disk_orig = disk_ifs[:,yy,xx]
f_bulge = fitted_bulge_ifs[:,yy,xx] * flux_unit
f_bulge_orig = bulge_ifs[:,yy,xx]
f_res = f_total - f_disk - f_bulge
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(decomp.wl, f_total, 'k', label='observado')
ax.plot(decomp.wl, f_res, 'm', label=u'resíduo')
ax.plot(decomp.wl, f_disk, 'b', label='disco (ajustado)')
ax.plot(decomp.wl, f_disk_orig, 'b:', label='disco (original)')
ax.plot(decomp.wl, f_bulge, 'r', label='bojo (ajustado)')
ax.plot(decomp.wl, f_bulge_orig, 'r:', label='bojo (original)')
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$F_\lambda\ [\mathrm{erg} / \mathrm{s} / \mathrm{cm}^2 / \mathrm{\AA}]$')
ax.set_title(r'$r = r_e$ ($%.1f^{\prime\prime}$)' % initial_model.bulge.r_e.value)

ax = plt.subplot(gs[2])
f_total = decomp.flux.sum(axis=2).sum(axis=1) * flux_unit
f_disk = fitted_disk_ifs.sum(axis=2).sum(axis=1) * flux_unit
f_disk_orig = disk_ifs.sum(axis=2).sum(axis=1)
f_bulge = fitted_bulge_ifs.sum(axis=2).sum(axis=1) * flux_unit
f_bulge_orig = bulge_ifs.sum(axis=2).sum(axis=1)
f_res = (decomp.flux - fitted_disk_ifs - fitted_bulge_ifs).sum(axis=2).sum(axis=1) * flux_unit
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(decomp.wl, f_total, 'k', label='observado')
ax.plot(decomp.wl, f_bulge, 'r', label='bojo (ajustado)')
ax.plot(decomp.wl, f_bulge_orig, 'r:', label='bojo (original)')
ax.plot(decomp.wl, f_disk, 'b', label='disco (ajustado)')
ax.plot(decomp.wl, f_disk_orig, 'b:', label='disco (original)')
ax.plot(decomp.wl, f_res, 'm', label=u'resíduo')
ax.set_xlim(decomp.wl.min(), decomp.wl.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xlabel(r'Comprimento de onda $[\AA]$')
ax.set_ylabel(r'$F_\lambda\ [\mathrm{erg} / \mathrm{s} / \mathrm{cm}^2 / \mathrm{\AA}]$')
ax.set_title(r'Integrado')

ax_in = fig.add_axes([0.66, 0.21, 0.3, 0.1])
l1 = find_nearest_index(wl, 6000.0)
l2 = find_nearest_index(wl, 6200.0)
ax_in.plot(decomp.wl[l1:l2], f_bulge[l1:l2], 'r')
ax_in.plot(decomp.wl[l1:l2], f_bulge_orig[l1:l2], 'r:')
ax_in.plot(decomp.wl[l1:l2], f_disk[l1:l2], 'b')
ax_in.plot(decomp.wl[l1:l2], f_disk_orig[l1:l2], 'b:')
#ax_in.xaxis.set_major_locator(MultipleLocator(500))
ax_in.set_xticks([])
ax_in.set_yticks([])

plt.suptitle(r'Espectros ajustados (PSF $FWHM = 2.9^{\prime\prime}$)')
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
plt.savefig('plots/tese/simulation_spectra.pdf')

################################################################################
##########
########## Fit quality
##########
################################################################################
fig = plt.figure(figsize=(width_in, 0.8 * width_in))
gs = plt.GridSpec(2, 1, height_ratios=[1.0, 1.0])

pa = (90.0 + initial_model.disk.PA.value) * np.pi / 180.0
ba = 1.0 - initial_model.disk.ell.value
disk_error = fitted_disk_ifs * flux_unit / disk_ifs
disk_error_mean = np.mean(disk_error)
disk_error_std = np.std(disk_error)
print 'Disk error: %.4f +- %.4f' % (disk_error_mean, disk_error_std)
disk_error_r = radialProfile(disk_error, bins, norm_x0, norm_y0, pa, ba, rad_scale=1.0)
disk_error_min, disk_error_max = np.percentile(disk_error_r, (5, 95))
disk_error_mean = np.mean(disk_error_r)
disk_error_std = np.std(disk_error_r)

pa = (90.0 + initial_model.bulge.PA.value) * np.pi / 180.0
ba = 1.0 - initial_model.bulge.ell.value
bulge_bins = np.arange(0.0, initial_model.bulge.r_e.value * 2.5 + 1.0)
bulge_error = fitted_bulge_ifs * flux_unit / bulge_ifs
good_bulge = distance(fitted_bulge_ifs.shape[1:], norm_x0, norm_y0, pa, ba) < (initial_model.bulge.r_e.value * 2.5)
print bulge_error.shape
bulge_error_mean = np.mean(bulge_error[:, good_bulge])
bulge_error_std = np.std(bulge_error[:, good_bulge])
print 'bulge error: %.4f +- %.4f' % (bulge_error_mean, bulge_error_std)
bulge_error_r = radialProfile(bulge_error, bulge_bins, norm_x0, norm_y0, pa, ba, rad_scale=1.0)
bulge_error_min, bulge_error_max = np.percentile(bulge_error_r, (5, 95))
bulge_error_mean = np.mean(bulge_error_r)
bulge_error_std = np.std(bulge_error_r)

error_range = np.abs(1.0 - np.array([bulge_error_min, bulge_error_max, disk_error_min, disk_error_max])).max()
vmin = 1.0 - error_range
vmax = 1.0 + error_range

ax = plt.subplot(gs[0])
im = ax.pcolormesh(wl, bulge_bins, bulge_error_r.T, vmin=vmin, vmax=vmax, cmap='RdBu')
#ax.text(0.1, 0.9, r'$%.4f\ \pm\ %.4f$' % (bulge_error_mean, bulge_error_std), transform=ax.transAxes)
ax.set_ylim(0, bulge_bins.max())
ax.set_xlim(wl.min(), wl.max())
ax.set_ylabel(r'Raio $[\mathrm{arcsec}]$')
ax.set_xticklabels([])
#ax.set_xlabel(r'Comprimento de onda $[\AA]$')
ax.set_title(u'Bojo')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[1])
im = ax.pcolormesh(wl, bins, disk_error_r.T, vmin=vmin, vmax=vmax, cmap='RdBu')
#ax.text(0.1, 0.9, r'$%.4f\ \pm\ %.4f$' % (disk_error_mean, disk_error_std), transform=ax.transAxes)
ax.set_ylim(0, bins.max())
ax.set_xlim(wl.min(), wl.max())
ax.set_ylabel(r'Raio $[\mathrm{arcsec}]$')
ax.set_xlabel(r'Comprimento de onda $[\AA]$')
ax.set_title(u'Disco')
plt.colorbar(im, ax=ax)

plt.suptitle(u'Razão ajuste / original (PSF $FWHM = 2.6^{\prime\prime}$)')
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
plt.savefig('plots/tese/simulation_error.pdf')


