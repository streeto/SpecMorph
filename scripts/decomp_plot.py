'''
Created on Jun 6, 2013

@author: andre
'''

# import matplotlib
# matplotlib.use('PDF')
import matplotlib.pyplot as plt
from pycasso.util import getPixelDistance
import numpy as np
import tables
import argparse

parser = argparse.ArgumentParser(description='Perform Bulge/Disk decomposition.')

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
box_radius = t.attrs.box_radius
fit_l_obs = grp.wavelength[:]

ylabel = {'I_Be': r'$\log I_{Be}\ [erg / s /cm^2 / \AA]$',
          'R_e': r'$R_e\ [arcsec]$',
          'I_D0': r'$\log I_{R0}\ [erg / s /cm^2 / \AA]$',
          'R_0': r'$R_0\ [arcsec]$',
          'sigma': r'$PSF_{FWHM}\ [arcsec]$',
          'R2': r'$R^2$',
          'x0': r'$X_{center}\ [pixel]$',
          'y0': r'$Y_{center}\ [pixel]$',
          'pa': r'$P.A.\ [degrees]$',
          'ba': r'$b/a$',
          }

nothing = lambda x: x
sigma_to_FWHM = lambda x: x * 2.0 * np.sqrt(2.0 * np.log(2.0))
rad_to_degrees = lambda x: x * 180.0 / np.pi
func = {'I_Be': np.log10,
        'R_e': nothing,
        'I_D0': np.log10,
        'R_0': nothing,
        'sigma': sigma_to_FWHM,
        'R2': nothing,
        'x0': nothing,
        'y0': nothing,
        'pa': rad_to_degrees,
        'ba': nothing,
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
            'image.cmap': 'OrRd',
            }
plt.rcParams.update(plotpars)

################################################################################
##########
########## All fit parameters 
##########
################################################################################
fig = plt.figure(1, figsize=(8, 10))
plt.clf()
fig.set_tight_layout(True)
gs = plt.GridSpec(5, 2)
for i, colname in enumerate(t.colnames):
    ax = plt.subplot(gs[i])
    y = func[colname](t.col(colname))
    ax.plot(fit_l_obs, y, 'k')
    ax.set_ylabel(ylabel[colname])
    if i >= 6:
        ax.set_xlabel(r'$wavelength\ [\AA]$')
    else:
        ax.set_xticklabels([])
plt.subplots_adjust(top=0.7)
plt.savefig('%s-%s-fit-parameters.pdf' % (galaxyId, args.decompId))


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
imshape = grp.bulge_spectra.shape[1:]
x0 = t.cols.x0[0]
y0 = t.cols.y0[0]
invalid = np.where(getPixelDistance(imshape, x0, y0) < 2.5)

bulge_im = grp.bulge_spectra[l1:l2].sum(axis=0)
bulge_im[invalid] = np.nan

disk_im = grp.disk_spectra[l1:l2].sum(axis=0)
disk_im[invalid] = np.nan

syn_im = grp.synth_spectra[l1:l2].sum(axis=0)
syn_im[invalid] = np.nan

vmin = -18.0
vmax = -16.0

ax = plt.subplot(gs[1,0])
im = ax.imshow(np.log10(bulge_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'Bulge model ($\log F_\lambda$)')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[1,1])
im = ax.imshow(np.log10(disk_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'Disk model ($\log F_\lambda$)')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,0])
im = ax.imshow(np.log10(syn_im), origin='lower', interpolation='nearest', vmin=vmin, vmax=vmax)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title(r'Synthetic image ($\log F_\lambda$)')
plt.colorbar(im, ax=ax)

ax = plt.subplot(gs[2,1])
im = ax.imshow(syn_im - disk_im - bulge_im, origin='lower', interpolation='nearest', vmin=-1e-18, vmax=1e-18)
ax.set_yticks([])
ax.set_xticks([])
ax.set_title('Residual')
plt.colorbar(im, ax=ax)

plt.suptitle(r'Model images @ $5635\ \AA$')
plt.savefig('%s-%s-model-images.pdf' % (galaxyId, args.decompId))



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
f_syn = grp.synth_spectra[:,37,37]
f_disk = grp.disk_spectra[:,37,37]
f_bulge = grp.bulge_spectra[:,37,37]
ax.plot(fit_l_obs, f_syn, 'g', label='original')
ax.plot(fit_l_obs, f_disk, 'b', label='disk')
ax.plot(fit_l_obs, f_bulge, 'r', label='bulge')
ax.plot(fit_l_obs, f_syn - f_disk - f_bulge, 'm', label='residual')
ax.set_xlabel(r'$wavelength\ [\AA]$')
ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
ax.text(0.1, 0.9, r'$R\ =\ 5\ arcsec$ | $\Delta\lambda\ =\ %d\ \AA$' % (4*box_radius+2), transform=ax.transAxes)
ax.legend()

ax = plt.subplot(gs[2])
I_Be = func['I_Be'](t.col('I_Be'))
I_D0 = func['I_D0'](t.col('I_D0'))
ax.plot(fit_l_obs, I_D0, 'b', label=r'Disk ($I_{D0}$)')
ax.plot(fit_l_obs, I_Be, 'r', label=r'Bulge ($I_{Re}$)')
ax.set_xlabel(r'$wavelength\ [\AA]$')
ax.set_ylabel(r'$\log\ intensity$')
ax.legend()

ax = plt.subplot(gs[3])
R_e = func['R_e'](t.col('R_e'))
R_0 = func['R_0'](t.col('R_0'))
ax.plot(fit_l_obs, R_0, 'b', label=r'Disk ($R_{0}$)')
ax.plot(fit_l_obs, R_e, 'r', label=r'Bulge ($R_{e}$)')
ax.set_xlabel(r'$wavelength\ [\AA]$')
ax.set_ylabel(r'$radius [arcsec]$')
ax.legend()
plt.savefig('%s-%s-model-quality.pdf' % (galaxyId, args.decompId))


db.close()
