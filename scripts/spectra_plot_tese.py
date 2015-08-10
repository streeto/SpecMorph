# -*- coding: utf-8 -*-
'''
Created on Jun 6, 2013

@author: andre
'''

from specmorph.io import DecompContainer
from specmorph.model import BDModel, create_model_images
from specmorph.geometry import ellipse_params
from imfit.psf import moffat_psf
from pycasso.util import radialProfile, getEllipseParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import numpy as np
import argparse
from os import path
from specmorph.util import find_nearest_index

width_pt = 448.07378
width_in = width_pt / 72.0 * 0.95

################################################################################
def getMinMax(image):
    vals = np.ma.masked_invalid(image).compressed()
    mean = vals.mean()
    sigma = np.sqrt(vals.var())
    return mean - 3 * sigma, mean + 3 * sigma
################################################################################


################################################################################
def load_line_mask(line_file, wl):
    import atpy
    import pystarlight.io  # @UnusedImport
    t = atpy.Table(line_file, type='starlight_mask')
    masked_wl = np.zeros(wl.shape, dtype='bool')
    for i in xrange(len(t)):
        l_low, l_upp, line_w, _ = t[i]
        if line_w > 0.0: continue
        masked_wl |= (wl > l_low) & (wl < l_upp)
    return masked_wl
################################################################################


################################################################################
def plot_setup(plot_file):
    pdf = PdfPages(plot_file)
    plotpars = {'legend.fontsize': 8,
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
    return pdf
################################################################################

parser = argparse.ArgumentParser(description='Plot Bulge/Disk decomposition.')

parser.add_argument('galaxyId', type=str, nargs=1,
                    help='CALIFA galaxy ID. Ex.: K0001')
parser.add_argument('--sample', dest='sample', default= 'sample004',
                    help='Sample file. Default: sample004.')
parser.add_argument('--db', dest='db', default='data/decomposition.005.h5',
                    help='HDF5 database path.')

args = parser.parse_args()
galaxyId = args.galaxyId[0]
sampleId = path.basename(args.sample)

c = DecompContainer()
c.loadHDF5(args.db, sampleId, galaxyId)
t = c.fitParams
t_1p = c.firstPassParams
l_obs_1p = t_1p['wl'][:]
flag_bad_1p = (t_1p['flag'] > 0.0) | (t_1p['chi2'] > 20.0)

box_radius = c.attrs['box_radius']
target_vd = c.attrs['target_vd']
PSF_FWHM = c.attrs['PSF_FWHM']
galaxyName = c.attrs['object_name']
flux_unit = c.attrs['flux_unit']
flag_bad = (t['flag'] > 0.0) | (t['chi2'] > 20.0)
flag_ok = ~flag_bad
fit_l_obs = t['wl']
masked_wl = load_line_mask('data/starlight/Mask.mE', fit_l_obs)

initial_params = np.array(c.initialParams, dtype=t.dtype)
initial_model = BDModel.fromParamVector(initial_params)
psf = moffat_psf(PSF_FWHM, c.attrs['PSF_beta'], size=c.attrs['PSF_size'])
bulge_model_im, disk_model_im = create_model_images(initial_model, c.total.f_obs.shape[1:], psf) 
bulge_model_im *= flux_unit
disk_model_im *= flux_unit

l1 = find_nearest_index(fit_l_obs, 5635.0)
x0 = t['x0'][0]
y0 = t['y0'][0]

total_im = c.total.f_obs[l1] * flux_unit
model_im = disk_model_im + bulge_model_im
residual_im = (total_im - model_im)  / total_im

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

disk_colnames = [
                 'I_0',
                 'h',
                 'PA_d',
                 'ell_d',
            ]

limits = {'I_e': (-17.5, -15),
          'r_e': (0, 5),
          'PA_b': (150, 180),
          'ell_b': (0.3, 0.5),
          'I_0': (-17.5, -15),
          'h': (0, 20),
          'PA_d': (150, 180),
          'ell_d': (0.3, 0.5),
          'x0': None,
          'y0': None,
          'chi2': (0,7),
          'n_pix': None,
          'n': (1, 4),
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
log10flux = lambda x: np.log10(x*flux_unit)
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

pdf = plot_setup('plots/%s_%s.pdf' % (galaxyId, sampleId))

################################################################################
##########
########## Initial model
##########
################################################################################
vmin, vmax = getMinMax(np.log10(total_im))
fig = plt.figure(figsize=(width_in, 0.8 * width_in))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
ax = plt.subplot(gs[0,0])
im = ax.imshow(np.log10(total_im), vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Observado')

ax = plt.subplot(gs[0,1])
im = ax.imshow(np.log10(model_im), vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Modelo')

#residual_range = np.abs(residual_im).max()
residual_range = 0.5
res_vmin = - residual_range
res_vmax = residual_range

ax = plt.subplot(gs[0,2])
im = ax.imshow(residual_im, vmin=res_vmin, vmax=res_vmax, cmap='RdBu')
plt.colorbar(im, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(u'Resíduo')


ax = plt.subplot(gs[1,:])
bins = np.arange(0, 32)
bins_c = bins[:-1]
y0 = initial_model.y0.value - 1
x0 = initial_model.x0.value - 1
pa_i, ell_i = ellipse_params(total_im, x0, y0)
pa_i = (90.0 + pa_i) * np.pi / 180.0
ba_i = 1.0 - ell_i

tr = radialProfile(np.log10(total_im), bins, x0, y0, pa_i, ba_i)
mr = radialProfile(np.log10(model_im), bins, x0, y0, pa_i, ba_i)
br = radialProfile(np.log10(bulge_model_im), bins, x0, y0, pa_i, ba_i)
dr = radialProfile(np.log10(disk_model_im), bins, x0, y0, pa_i, ba_i)

ax.plot(bins_c, tr, 'k-', label='Observado')
ax.plot(bins_c, mr, 'k--', label='Modelo')
ax.plot(bins_c, br, 'r-', label='Bojo')
ax.plot(bins_c, dr, 'b-', label='Disco')
ax.set_xlabel(r'Raio $[\mathrm{arcsec}]$')
ax.set_ylabel(r'$\log$ Fluxo')
ax.set_xlim(0.0, 20.0)
ax.set_ylim(-17.5, -15.5)
ax.legend(loc='upper right', frameon=False)

plt.suptitle(r'Modelo ajustado em $5635\,\mathrm{\AA}$ - %s (%s)' % (galaxyId, galaxyName))
gs.tight_layout(fig, rect=[0, 0, 1, 0.95])

pdf.savefig()

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
    y = np.ma.array(func[colname](t[colname]), mask=flag_bad)
    l = np.ma.array(fit_l_obs, mask=flag_bad)
    y_1p = np.ma.array(func[colname](t_1p[colname]), mask=flag_bad_1p)
    l_1p = np.ma.array(l_obs_1p, mask=flag_bad_1p)
    ax.plot(l, y, 'k')
    ax.plot(l_1p, y_1p, '.b')
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
    colors = np.where(masked_wl[flag_bad], 'gray', 'pink')
    ax.vlines(fit_l_obs[flag_bad], ymin, ymax, colors, alpha=0.25)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(fit_l_obs.min(), fit_l_obs.max())
plt.suptitle(u'Parâmetros morfológicos - %s (%s)' % (galaxyId, galaxyName))
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])

pdf.savefig(fig)


################################################################################
##########
########## Model images
##########
################################################################################
fig = plt.figure(figsize=(width_in, 0.8 * width_in))
gs = plt.GridSpec(3, 2, height_ratios=[-0.2, 1.0, 1.0])

vmin, vmax = getMinMax(np.log10(total_im))
res_vmin, res_vmax = getMinMax(residual_im)
fit_type = 'obs'

ax = plt.subplot(gs[1,0])
im = ax.imshow(np.log10(bulge_model_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'$\log F\ (\mathrm{bojo})$')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[1,1])
im = ax.imshow(np.log10(disk_model_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'$\log F\ (\mathrm{disco})$')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,0])
im = ax.imshow(np.log10(total_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'$\log F\ (\mathrm{observado}$)')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,1])
im = ax.imshow(residual_im, origin='lower', interpolation='nearest', cmap='RdBu', vmin=res_vmin, vmax=res_vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(u'$\Delta F /F\ (\mathrm{resíduo}$)')
plt.colorbar(im, ax=ax)

plt.suptitle(u'Componentes morfológicas em $5635\,\mathrm{\AA}$ - %s (%s)' % (galaxyId, galaxyName))
gs.tight_layout(fig, rect=[0, 0, 1, 0.9])
pdf.savefig(fig)


################################################################################
##########
########## Spectra
##########
################################################################################
fig = plt.figure(figsize=(width_in, 1.3 * width_in))
gs = plt.GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.0])

ax = plt.subplot(gs[0])
xx = np.round(initial_model.x0.value)
yy = np.round(initial_model.y0.value)
f_total = c.total.f_obs[:,yy,xx] * flux_unit
f_disk = c.disk.f_obs[:,yy,xx] * flux_unit
f_bulge = c.bulge.f_obs[:,yy,xx] * flux_unit
f_res = f_total - f_disk - f_bulge
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(fit_l_obs, f_total, 'k', label='observado')
ax.plot(fit_l_obs, f_res, 'm', label=u'resíduo')
ax.plot(fit_l_obs, f_disk, 'b', label='disco')
ax.plot(fit_l_obs, f_bulge, 'r', label='bojo')
ax.plot(fit_l_obs, np.zeros_like(fit_l_obs), 'k:')
ax.set_xlim(fit_l_obs.min(), fit_l_obs.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$F_\lambda\ [\mathrm{erg} / \mathrm{s} / \mathrm{cm}^2 / \mathrm{\AA}]$')
ax.set_title(r'Nuclear')
ax.legend(loc='center right', frameon=False)

ax = plt.subplot(gs[1])
xx = np.ceil(initial_model.x0.value + initial_model.bulge.r_e.value)
f_total = c.total.f_obs[:,yy,xx] * flux_unit
f_disk = c.disk.f_obs[:,yy,xx] * flux_unit
f_bulge = c.bulge.f_obs[:,yy,xx] * flux_unit
f_res = f_total - f_disk - f_bulge
f_res = f_total - f_disk - f_bulge
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(fit_l_obs, f_total, 'k', label='observado')
ax.plot(fit_l_obs, f_res, 'm', label=u'resíduo')
ax.plot(fit_l_obs, f_disk, 'b', label='disco')
ax.plot(fit_l_obs, f_bulge, 'r', label='bojo')
ax.plot(fit_l_obs, np.zeros_like(fit_l_obs), 'k:')
ax.set_xlim(fit_l_obs.min(), fit_l_obs.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$F_\lambda\ [\mathrm{erg} / \mathrm{s} / \mathrm{cm}^2 / \mathrm{\AA}]$')
ax.set_title(r'$r = r_e$ ($%.1f^{\prime\prime}$)' % initial_model.bulge.r_e.value)

ax = plt.subplot(gs[2])
f_total = c.total.i_f_obs * flux_unit
f_disk = c.disk.i_f_obs * flux_unit
f_bulge = c.bulge.i_f_obs * flux_unit
f_res = (f_total - f_disk - f_bulge) * flux_unit
vmin = min(f_total.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(fit_l_obs, f_total, 'k', label='observado')
ax.plot(fit_l_obs, f_res, 'm', label=u'resíduo')
ax.plot(fit_l_obs, f_disk, 'b', label='disco')
ax.plot(fit_l_obs, f_bulge, 'r', label='bojo')
ax.plot(fit_l_obs, np.zeros_like(fit_l_obs), 'k:')
ax.set_xlim(fit_l_obs.min(), fit_l_obs.max())
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xlabel(r'Comprimento de onda $[\AA]$')
ax.set_ylabel(r'$F_\lambda\ [\mathrm{erg} / \mathrm{s} / \mathrm{cm}^2 / \mathrm{\AA}]$')
ax.set_title(r'Integrado')

plt.suptitle(u'Espectros das componentes morfológicas - %s (%s)' % (galaxyId, galaxyName))
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])

pdf.savefig(fig)

################################################################################
##########
########## Radial profiles
##########
################################################################################
fig = plt.figure(figsize=(width_in, 1.3 * width_in))
N_cols = 4
N_rows = 5
N_cell = N_cols * N_rows
delta_l = len(fit_l_obs) / N_cell
l_bins = np.linspace(50, len(fit_l_obs) - 51, N_cell, dtype='int')
gs = plt.GridSpec(N_rows, N_cols)
bin_r = np.arange(30)
bin_c = bin_r[:-1] + 0.5

for i in xrange(N_cols):
    for j in xrange(N_rows):
        ax = plt.subplot(gs[j, i])
        cell = j * N_cols + i
        l1 = l_bins[cell]
        if (cell + 1) < N_cell:
            l2 = l_bins[cell + 1] 
        else:
            l2 = len(fit_l_obs) - 1
        l = l1
        while l < l2:
            if t['flag'][l] == 0:
                break
            l += 1
        else:
            print 'Only flagged stuff in the interval [%d:%d]' % (fit_l_obs[l1], fit_l_obs[l2])
            l = l1
        wl = fit_l_obs[l]
        x0 = t['x0'][l] - 1
        y0 = t['y0'][l] - 1
        bulge_im = c.bulge.f_obs[l] * flux_unit
        disk_im = c.disk.f_obs[l] * flux_unit
        total_im = c.total.f_obs[l] * flux_unit
        mask = ~np.isnan(total_im)
        pa, ba = getEllipseParams(total_im, x0, y0, mask=mask)
        bulge_r = radialProfile(bulge_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        disk_r = radialProfile(disk_im, bin_r, x0, y0, pa, ba, rad_scale=1.0)
        total_r = radialProfile(total_im, bin_r, x0, y0, pa, ba, rad_scale=1.0, mask=mask)
        ax.plot(bin_c, np.log10(total_r), 'k', label='observed')
        ax.plot(bin_c, np.log10(disk_r + bulge_r), 'k:', label='model')
        ax.plot(bin_c, np.log10(disk_r), 'b:', label='disk model')
        ax.plot(bin_c, np.log10(bulge_r), 'r:', label='bulge model')
        ax.text(0.5, 0.85, r'$%d\ \AA$' % wl, transform=ax.transAxes)
        ax.set_ylim(-18, -15.5)
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
