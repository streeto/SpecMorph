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
flag_bad_1p = (t_1p['flag'] > 0.0) | (t_1p['chi2'] > 10.0)

box_radius = c.attrs['box_radius']
target_vd = c.attrs['target_vd']
PSF_FWHM = c.attrs['PSF_FWHM']
galaxyName = c.attrs['object_name']
flux_unit = c.attrs['flux_unit']
flag_bad = (t['flag'] > 0.0) | (t['chi2'] > 2.0)
flag_ok = ~flag_bad
fit_l_obs = t['wl']
masked_wl = load_line_mask('data/starlight/Mask.mE', fit_l_obs)

initial_params = np.array(c.initialParams, dtype=t.dtype)
initial_model = BDModel.fromParamVector(initial_params)
psf = moffat_psf(PSF_FWHM, c.attrs['PSF_beta'], size=c.attrs['PSF_size'])
bulge_model_im, disk_model_im = create_model_images(initial_model, c.total.f_obs.shape[1:], psf) 
bulge_model_im *= flux_unit
disk_model_im *= flux_unit

l_range = np.where((fit_l_obs > 5590.0) & (fit_l_obs < 5680.0))[0]
l1 = l_range[0]
l2 = l_range[-1]
x0 = t['x0'][0]
y0 = t['y0'][0]

bulge_im = np.median(c.bulge.f_obs[l1:l2], axis=0) * flux_unit
disk_im = np.median(c.disk.f_obs[l1:l2], axis=0) * flux_unit
total_im = np.median(c.total.f_obs[l1:l2], axis=0) * flux_unit
residual_im = (total_im - disk_im - bulge_im)  / total_im

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

disk_colnames = [
                 'I_0',
                 'h',
                 'PA_d',
                 'ell_d',
            ]

limits = {'I_e': (-18, -16),
          'r_e': (0, 35),
          'PA_b': (-5, 185),
          'ell_b': (-0.05, 1.0),
          'I_0': (-18, -16),
          'h': (0, 35),
          'PA_d': (-5, 185),
          'ell_d': (-0.05, 1.05),
          'x0': None,
          'y0': None,
          'chi2': (0,7),
          'n_pix': None,
          'n': (0.95, 6.05),
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

pdf = plot_setup('plots/%s_%s.pdf' % (galaxyId, sampleId))

################################################################################
##########
########## Initial model
##########
################################################################################
vmin, vmax = getMinMax(np.log10(total_im))
fig = plt.figure(1, figsize=(8, 6))
gs = plt.GridSpec(2, 3, height_ratios=[2.0, 3.0])
ax = plt.subplot(gs[0,0])
im = ax.imshow(np.log10(total_im), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Total')
plt.colorbar(im, ax=ax)


ax = plt.subplot(gs[0,1])
im = ax.imshow(np.log10(bulge_model_im), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Bulge')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[0,2])
im = ax.imshow(np.log10(disk_model_im), vmin=vmin, vmax=vmax)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(r'Disk')
plt.colorbar(im, ax=ax)
    
ax = plt.subplot(gs[1,:])
bins = np.arange(0, 32)
bins_c = bins[:-1] + 0.5
y0 = initial_model.y0.value
x0 = initial_model.x0.value
pa_i, ell_i = ellipse_params(total_im, x0, y0)
pa_i = (90.0 + pa_i) * np.pi / 180.0
ba_i = 1.0 - ell_i
mr = radialProfile(np.log10(total_im), bins, x0, y0, pa_i, ba_i)
br = radialProfile(np.log10(bulge_model_im), bins, x0, y0, pa_i, ba_i)
dr = radialProfile(np.log10(disk_model_im), bins, x0, y0, pa_i, ba_i)
ax.plot(bins_c, mr, 'k-', label='Total')
ax.plot(bins_c, br, 'r-', label='Bulge')
ax.plot(bins_c, dr, 'b-', label='Disk')
ax.set_xlabel(r'Radius [arcsec]')
ax.set_ylabel(r'$\log$ flux (relative)')
ax.set_xlim(0.0, 30.0)
ax.set_ylim(mr.min(), mr.max())
ax.legend(loc='upper right')

tmp = (initial_model.bulge.I_e.value,
       initial_model.bulge.r_e.value,
       initial_model.bulge.n.value,
       initial_model.disk.I_0.value,
       initial_model.disk.h.value,
       PSF_FWHM)
plt.suptitle(r'Initial model: $I_e = %.3f$, $r_e = %.3f$, $n = %.3f$, $I_0 = %.3f$, $h = %.3f$, $FWHM = %.2f$' % tmp)
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig()

################################################################################
##########
########## All fit parameters 
##########
################################################################################
fig = plt.figure(2, figsize=(8, 7))
n_rows = 4
n_cols = 3
gs = plt.GridSpec(n_rows, n_cols)
for i, colname in enumerate(colnames):
    if colname is None: continue
    ax = plt.subplot(gs[i])
    y = np.ma.array(func[colname](t[colname]), mask=flag_bad)
    l = np.ma.array(fit_l_obs, mask=flag_bad)
    y_1p = np.ma.array(func[colname](t_1p[colname]), mask=flag_bad_1p)
    l_1p = np.ma.array(l_obs_1p, mask=flag_bad_1p)
    ax.plot(l_1p, y_1p, '.b')
    ax.plot(l, y, 'k')
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
        ymin = min(y.min(), y_1p.min())
        ymax = max(y.max(), y_1p.max())
    colors = np.where(masked_wl[flag_bad], 'gray', 'pink')
    ax.vlines(fit_l_obs[flag_bad], ymin, ymax, colors, alpha=0.25)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(fit_l_obs.min(), fit_l_obs.max())
plt.suptitle('%s - %s' % (galaxyName, galaxyId))
gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig(fig)


################################################################################
##########
########## Model images
##########
################################################################################
fig = plt.figure(3, figsize=(8, 6))
gs = plt.GridSpec(3, 2, height_ratios=[-0.2, 1.0, 1.0])



vmin, vmax = getMinMax(np.log10(total_im))
res_vmin, res_vmax = getMinMax(residual_im)
fit_type = 'obs'

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
ax.set_title(r'$\log F^{%s}_\lambda$' % fit_type)
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,1])
im = ax.imshow(residual_im, origin='lower', interpolation='nearest', cmap='RdBu', vmin=res_vmin, vmax=res_vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('$(F^{%s}_\lambda - F^{bulge}_\lambda - F^{disk}_\lambda) / F^{%s}_\lambda$' % (fit_type, fit_type))
plt.colorbar(im, ax=ax)

plt.suptitle(r'%s - %s | Model images @ $5635\ \AA$, PSF FWHM = $%.1f$ arcsec' % (galaxyName, galaxyId, PSF_FWHM))
gs.tight_layout(fig, rect=[0, 0, 1, 0.9])
pdf.savefig(fig)


################################################################################
##########
########## Fit quality
##########
################################################################################
fig = plt.figure(4, figsize=(8, 10))
gs = plt.GridSpec(2, 1, height_ratios=[1.0, 1.0])
ax = plt.subplot(gs[0])

xx = np.round(initial_model.x0.value)
yy = np.round(initial_model.y0.value)

f_total = c.total.f_obs[:,yy,xx] * flux_unit
f_disk = c.disk.f_obs[:,yy,xx] * flux_unit
f_bulge = c.bulge.f_obs[:,yy,xx] * flux_unit
f_res = f_total - f_disk - f_bulge
vmax = 1.05 * max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
vmin = -2.0 * f_res.std()

ax.plot(fit_l_obs, f_total, 'k', label='observed')
ax.plot(fit_l_obs, f_res, 'm', label='residual')
ax.plot(fit_l_obs, f_disk, 'b', label='disk model')
ax.plot(fit_l_obs, f_bulge, 'r', label='bulge model')
ax.set_xlim(fit_l_obs.min(), fit_l_obs.max())
ax.set_ylim(vmin, vmax)
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xticklabels([])
ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
ax.set_title(r'Spectra at the nucleus')
ax.legend()

ax = plt.subplot(gs[1])
xx = np.ceil(initial_model.x0.value + initial_model.bulge.r_e.value)
f_total = c.total.f_obs[:,yy,xx] * flux_unit
f_disk = c.disk.f_obs[:,yy,xx] * flux_unit
f_bulge = c.bulge.f_obs[:,yy,xx] * flux_unit
f_res = f_total - f_disk - f_bulge
vmax = 1.05 * max(f_total.max(), f_disk.max(), f_bulge.max(), f_res.max())
vmin = -2.0 * f_res.std()

ax.plot(fit_l_obs, f_total, 'k', label='observed')
ax.plot(fit_l_obs, f_res, 'm', label='residual')
ax.plot(fit_l_obs, f_disk, 'b', label='disk model')
ax.plot(fit_l_obs, f_bulge, 'r', label='bulge model')
ax.set_xlim(fit_l_obs.min(), fit_l_obs.max())
ax.set_ylim(vmin, vmax)
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.set_xlabel(r'wavelength $[\AA]$')
ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
ax.set_title(r'Spectra at $R = r_e$ ($%.1f\,arcsec$)' % initial_model.bulge.r_e.value)

gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
pdf.savefig(fig)

################################################################################
##########
########## Radial profiles
##########
################################################################################
fig = plt.figure(5, figsize=(8, 10))
N_cols = 5
N_rows = 6
N_cell = N_cols * N_rows
delta_l = len(fit_l_obs) / N_cell
l_bins = np.arange(0, len(fit_l_obs), delta_l)
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
        x0 = t['x0'][l]
        y0 = t['y0'][l]
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
