'''
Created on Jun 6, 2013

@author: andre
'''

import matplotlib
from pycasso.util import getImageDistance, radialProfile, getEllipseParams
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import numpy as np
import tables
import argparse

parser = argparse.ArgumentParser(description='Plot Bulge/Disk decomposition.')

parser.add_argument('galaxyId', type=str, nargs=1,
                    help='CALIFA galaxy ID. Ex.: K0001')
parser.add_argument('runId', type=str, nargs=1,
                    help='runId string. Ex.: eBR_v20.d13c500.ps03.k1.m0.CCM.Bgsd01.v01')
parser.add_argument('--decomp-id', dest='decompId', default= 'decomposition',
                    help='Decomposition label.')
parser.add_argument('--db', dest='db', default='decomposition.005.h5',
                    help='HDF5 database path.')

args = parser.parse_args()
galaxyId = args.galaxyId[0]
# HACK: remove the dots in run name.
runId = args.runId[0]
groupname = runId.replace('.', '_')

db = tables.openFile(args.db, 'r')
grp = db.getNode('/%s/%s/%s' % (args.decompId, groupname, galaxyId))
t = grp.fit_parameters
has_1p = False
if 'first_pass_parameters' in grp:
    has_1p = True
    t_1p = grp.first_pass_parameters
    l_obs_1p = grp.first_pass_l_obs[:]
    flag_bad_1p = t_1p.cols.flag[:] > 0.0

box_radius = t.attrs.box_radius
FWHM = t.attrs.FWHM
galaxyName = t.attrs.object_name
flux_unit = t.attrs.flux_unit
flag_ok = (t.cols.flag[:] == 0.0)
flag_bad = (t.cols.flag[:] > 0.0)
fit_l_obs = grp.l_obs[:]

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

limits = {'I_e': (-18, -16),
          'r_e': (0, 20),
          'PA_b': (0, 40),
          'ell_b': (0.6, 0.9),
          'I_0': (-18, -16),
          'h': (0, 20),
          'PA_d': (0, 40),
          'ell_d': (0.6, 0.9),
          'x0': None,
          'y0': None,
          'chi2': None,
          'n_pix': None,
          'n': (2,4),
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

plotpars = {'legend.fontsize': 8,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'text.fontsize': 10,
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

################################################################################
##########
########## All fit parameters 
##########
################################################################################
fig = plt.figure(1, figsize=(8, 8))
plt.clf()
fig.set_tight_layout(True)
gs = plt.GridSpec(4, 3)
for i, colname in enumerate(colnames):
    if colname is None: continue
    ax = plt.subplot(gs[i])
    y = np.ma.array(func[colname](t.col(colname)), mask=flag_bad)
    l = np.ma.array(fit_l_obs, mask=flag_bad)
    if has_1p:
        y_1p = np.ma.array(func[colname](t_1p.col(colname)), mask=flag_bad_1p)
        l_1p = np.ma.array(l_obs_1p, mask=flag_bad_1p)
        ax.plot(l_1p, y_1p, '.r')
    ax.plot(l, y, 'k')
    ax.set_ylabel(ylabel[colname])
    ax.set_xlabel(r'wavelength $[\AA]$')
    if limits[colname] is not None:
        ymin = limits[colname][0]
        ymax = limits[colname][1]
    else:
        ymin = y.min()
        ymax = y.max()
    ax.vlines(fit_l_obs[flag_bad], ymin, ymax, 'gray', alpha=0.5)
    ax.set_ylim(ymin, ymax)
plt.subplots_adjust(top=0.7)
plt.suptitle('%s - %s' % (galaxyName, galaxyId))
plt.savefig('plots/%s_%s_%s-fit-parameters.pdf' % (galaxyId, runId, args.decompId))


################################################################################
##########
########## Model images
##########
################################################################################
fig = plt.figure(2, figsize=(8, 6))
plt.clf()
# fig.set_tight_layout(True)
gs = plt.GridSpec(3, 2, height_ratios=[-0.2, 1.0, 1.0])

l_range = np.where((fit_l_obs > 5590.0) & (fit_l_obs < 5680.0))[0]
l1 = l_range[0]
l2 = l_range[-1]
x0 = t.cols.x0[0]
y0 = t.cols.y0[0]

bulge_im = np.median(grp.f_syn_bulge__lyx[l1:l2], axis=0)

disk_im = np.median(grp.f_syn_disk__lyx[l1:l2], axis=0)

syn_im = np.median(grp.f_syn__lyx[l1:l2], axis=0)

residual_im = (syn_im - disk_im - bulge_im)  / syn_im


def getMinMax(image):
    vals = np.ma.masked_invalid(image).compressed()
    mean = vals.mean()
    sigma = np.sqrt(vals.var())
    return mean - 3 * sigma, mean + 3 * sigma


vmin, vmax = getMinMax(np.log10(syn_im))
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
im = ax.imshow(np.log10(syn_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'$\log F^{syn}_\lambda$')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,1])
im = ax.imshow(residual_im, origin='lower', interpolation='nearest', cmap='RdBu', vmin=res_vmin, vmax=res_vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('$(F^{syn}_\lambda - F^{bulge}_\lambda - F^{disk}_\lambda) / F^{syn}_\lambda$')
plt.colorbar(im, ax=ax)

plt.subplots_adjust(top=0.87)

plt.suptitle(r'%s - %s | Model images @ $5635\ \AA$, PSF FWHM = $%.1f$ arcsec' % (galaxyName, galaxyId, FWHM))
plt.savefig('plots/%s_%s_%s-model-images.pdf' % (galaxyId, runId, args.decompId))



################################################################################
##########
########## Fit quality
##########
################################################################################
fig = plt.figure(3, figsize=(8, 10))
plt.clf()
fig.set_tight_layout(True)
gs = plt.GridSpec(4, 1, height_ratios=[-0.2, 1.0, 1.0, 1.0])

ax = plt.subplot(gs[1])
l = np.ma.array(fit_l_obs, mask=flag_bad)
f_syn = np.ma.array(grp.f_syn__lyx[:,37,37], mask=flag_bad)
f_disk = np.ma.array(grp.f_syn_disk__lyx[:,37,37], mask=flag_bad)
f_bulge = np.ma.array(grp.f_syn_bulge__lyx[:,37,37], mask=flag_bad)
f_res = f_syn - f_disk - f_bulge
vmin = min(f_syn.min(), f_disk.min(), f_bulge.min(), f_res.min())
vmax = max(f_syn.max(), f_disk.max(), f_bulge.max(), f_res.max())
ax.plot(l, f_syn, 'k', label='synthetic')
ax.plot(l, f_disk, 'b', label='disk model')
ax.plot(l, f_bulge, 'r', label='bulge model')
ax.plot(l, f_res, 'm', label='residual')
ax.vlines(fit_l_obs[flag_bad], vmin, vmax, 'gray', alpha=0.5)
ax.set_xlabel(r'wavelength $[\AA]$')
ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
ax.text(0.1, 0.9, r'%s - %s | $R\ =\ 5\ arcsec$ | $\Delta\lambda\ =\ %d\ \AA$' % (galaxyName, galaxyId, 4*box_radius+2),
        transform=ax.transAxes)
ax.legend()

ax = plt.subplot(gs[2])
l = np.ma.array(fit_l_obs, mask=flag_bad)
I_e = np.ma.array(func['I_e'](t.col('I_e')), mask=flag_bad)
I_0 = np.ma.array(func['I_0'](t.col('I_0')), mask=flag_bad)
vmin = min(I_e.min(), I_0.min())
vmax = max(I_e.max(), I_0.max())
ax.plot(l, I_0, 'b', label=r'Disk $(I_{0})$')
ax.plot(l, I_e, 'r', label=r'Bulge $(I_{e})$')
if limits['I_0'] is not None:
    ymin = limits['I_0'][0]
    ymax = limits['I_0'][1]
else:
    ymin = y.min()
    ymax = y.max()
ax.vlines(fit_l_obs[flag_bad], ymin, ymax, 'gray', alpha=0.5)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r'wavelength $[\AA]$')
ax.set_ylabel(r'$\log\ I$')
ax.legend()

ax = plt.subplot(gs[3])
r_e = np.ma.array(func['r_e'](t.col('r_e')), mask=flag_bad)
h = np.ma.array(func['h'](t.col('h')), mask=flag_bad)
vmin = min(r_e.min(), h.min())
vmax = max(r_e.max(), h.max())
ax.plot(l, h, 'b', label=r'Disk $(h)$')
ax.plot(l, r_e, 'r', label=r'Bulge $(r_{e})$')
if limits['r_e'] is not None:
    ymin = limits['r_e'][0]
    ymax = limits['r_e'][1]
else:
    ymin = y.min()
    ymax = y.max()
ax.vlines(fit_l_obs[flag_bad], ymin, ymax, 'gray', alpha=0.5)
ax.set_ylim(ymin, ymax)
ax.set_xlabel(r'wavelength $[\AA]$')
ax.set_ylabel(r'radius $[arcsec]$')
ax.legend()
plt.savefig('plots/%s_%s_%s-model-quality.pdf' % (galaxyId, runId, args.decompId))

################################################################################
##########
########## Radial profiles
##########
################################################################################
fig = plt.figure(4, figsize=(8, 10))
plt.clf()
fig.set_tight_layout(True)
N_col = 5
N_row = 6
N_cell = N_col * N_row
delta_l = len(fit_l_obs) / N_cell
l_bins = np.arange(0, len(fit_l_obs), delta_l)
gs = plt.GridSpec(N_row, N_col)
bin_r = np.arange(30)
bin_c = bin_r[:-1] + 0.5

for i in xrange(N_col):
    for j in xrange(N_row):
        ax = plt.subplot(gs[j, i])
        cell = j * N_col + i
        l1 = l_bins[cell]
        if cell < N_cell:
            l2 = l_bins[cell+1]
        else:
            l2 = len(fit_l_obs) - 1
        l = l1
        while l < l2:
            if t.cols.flag[l] == 0:
                break
            l += 1
        else:
            print 'Only flagged stuff in the interval [%d:%d]' % (fit_l_obs[l1], fit_l_obs[l2])
            l = l1
        wl = fit_l_obs[l]
        x0 = t.cols.x0[l]
        y0 = t.cols.y0[l]
        bulge_im = grp.f_syn_bulge__lyx[l]
        disk_im = grp.f_syn_disk__lyx[l]
        syn_im = grp.f_syn__lyx[l]
        pa, ba = getEllipseParams(syn_im, x0, y0)
        r__yx = getImageDistance(syn_im.shape, x0, y0, pa, ba)
        bulge_r = radialProfile(bulge_im, r__yx, bin_r, rad_scale=1.0)
        disk_r = radialProfile(disk_im, r__yx, bin_r, rad_scale=1.0)
        syn_r = radialProfile(syn_im, r__yx, bin_r, rad_scale=1.0)
        
        ax.plot(bin_c, np.log10(syn_r), 'k', label='synthetic')
        ax.plot(bin_c, np.log10(disk_r + bulge_r), 'k:', label='model')
        ax.plot(bin_c, np.log10(disk_r), 'b:', label='disk model')
        ax.plot(bin_c, np.log10(bulge_r), 'r:', label='bulge model')
        ax.text(0.5, 0.85, r'$%d\ \AA$' % wl, transform=ax.transAxes)
        ax.set_ylim(-18, -15.5)
        if i == 0 and j == (N_row - 1):
            ax.set_xlabel(r'radius $[arcsec]$')
            ax.set_ylabel(r'$\log F_\lambda\ [erg / s / cm^2 / \AA]$')
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

plt.savefig('plots/%s_%s_%s-radial-profile.pdf' % (galaxyId, runId, args.decompId))


db.close()
