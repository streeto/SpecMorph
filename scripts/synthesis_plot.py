
'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
import numpy as np
import argparse
from os import path
from specmorph.califa.tables import califa_id_from_cube


################################################################################
def getMinMax(image):
    vals = np.ma.masked_invalid(image).compressed()
    mean = vals.mean()
    sigma = np.sqrt(vals.var())
    return mean - 3 * sigma, mean + 3 * sigma
################################################################################


################################################################################
def load_sample(fname):
    from astropy.io import ascii
    return ascii.read(fname, Reader=ascii.CommentedHeader)
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
def load_base(base_file):
    import atpy
    import pystarlight.io  # @UnusedImport
    t = atpy.Table(base_file, type='starlightv4_base')
    return t
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


################################################################################
def getCube(root, galaxyId, sampleId, component, HLR=None):
    filename = '%s_synthesis_%s_%s.fits' % (galaxyId, sampleId, component)
    filepath = path.join(root, filename)
    K = fitsQ3DataCube(filepath)
    pa, ba = K.getEllipseParams()
    K.setGeometry(pa, ba, HLR)
    #K.qMask &= K.filterResidual__yx()
    #K.qMask &= (K.zoneToYX(K.adevS, extensive=False) < 10.0)
    return K
################################################################################


################################################################################
def getArea(K, radius):
    radius_HLR = K.getHLR_pix() * radius
    area_pix = (K.qMask & (K.pixelDistance__yx < radius_HLR)).sum()
    return area_pix * K.parsecPerPixel**2
################################################################################
    

################################################################################
def plotall(gal, sampleId, args, pdf):
    galaxyId = califa_id_from_cube(gal['cube'])

    try:
        Kt = getCube(args.cubeDir, galaxyId, sampleId, 'total')
        Kb = getCube(args.cubeDir, galaxyId, sampleId, 'bulge', Kt.HLR_pix)
        Kd = getCube(args.cubeDir, galaxyId, sampleId, 'disk', Kt.HLR_pix)
    except:
        print '***** Skipped galaxy %s' % galaxyId
        return
    
    base = load_base(args.basePath)
    
    ################################################################################
    ##########
    ########## Spectra and residuals
    ##########
    ################################################################################
    fig = plt.figure(1, figsize=(8, 6))
    gs = plt.GridSpec(2, 1, height_ratios=[3.0, 1.0])
    ax = plt.subplot(gs[0])
    
    fobs_T = np.ma.array(Kt.integrated_f_obs, mask=Kt.integrated_f_flag > 0.0)
    fobs_B = np.ma.array(Kb.integrated_f_obs, mask=Kb.integrated_f_flag > 0.0)
    fobs_D = np.ma.array(Kd.integrated_f_obs, mask=Kd.integrated_f_flag > 0.0)
    
    fsyn_T = Kt.integrated_f_syn
    fsyn_B = Kb.integrated_f_syn
    fsyn_D = Kd.integrated_f_syn
    
    fres_T = (fobs_T - fsyn_T) / fobs_T * 100.0
    fres_B = (fobs_B - fsyn_B) / fobs_B * 100.0
    fres_D = (fobs_D - fsyn_D) / fobs_D * 100.0
    
    ax.plot(Kt.l_obs, fobs_T, 'k', label='Original')
    ax.plot(Kt.l_obs, fsyn_T, 'k:', label='Original (syn)')
    ax.plot(Kb.l_obs, fobs_B, 'r', label='Bulge')
    ax.plot(Kt.l_obs, fsyn_B, 'r:', label='Bulge (syn)')
    ax.plot(Kd.l_obs, fobs_D, 'b', label='Disk')
    ax.plot(Kt.l_obs, fsyn_D, 'b:', label='Disk (syn)')
    
    ax.set_xlim(Kt.l_obs.min(), Kt.l_obs.max())
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.set_xticklabels([])
    ax.set_ylabel(r'$F_\lambda\ [erg / s / cm^2 / \AA]$')
    ax.set_title(r'Integrated spectra')
    ax.legend()
    
    ax = plt.subplot(gs[1])
    ax.plot(Kt.l_obs, fres_T, 'k', label='Original')
    ax.plot(Kb.l_obs, fres_B, 'r', label='Bulge')
    ax.plot(Kd.l_obs, fres_D, 'b', label='Disk')
    ax.plot(Kd.l_obs, np.zeros_like(Kd.l_obs), 'k:')
    
    ax.set_xlim(Kt.l_obs.min(), Kt.l_obs.max())
    ax.set_ylim(-15.0, 15.0)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.set_xlabel(r'wavelength $[\AA]$')
    ax.set_ylabel(r'Residual [%]')
    ax.grid(True)
    
    m_B = Kb.integrated_Mcor / Kt.integrated_Mcor * 100.0
    m_D = Kd.integrated_Mcor / Kt.integrated_Mcor * 100.0
    l_B = Kb.integrated_Lobn / Kt.integrated_Lobn * 100.0
    l_D = Kd.integrated_Lobn / Kt.integrated_Lobn * 100.0
    
    plt.suptitle('%s - %s - Bgsd6e' % (galaxyId, gal['Name']))
    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    
    
    ################################################################################
    ##########
    ########## Radial profiles
    ##########
    ################################################################################
    radius_HLR = 2.5
    
    bins_T = np.arange(int(Kt.getHLR_pix() * radius_HLR))
    #bins_T = np.arange(30)
    binc_T = bins_T[:-1] + 0.5
    atf_T = Kt.radialProfile(Kt.at_flux__yx * Kt.LobnSD__yx, bins_T, rad_scale=1, mode='sum') / \
                Kt.radialProfile(Kt.LobnSD__yx, bins_T, rad_scale=1, mode='sum') 
    AV_T = Kt.radialProfile(Kt.A_V__yx, bins_T, rad_scale=1)
    azm_T = Kt.radialProfile(Kt.alogZ_mass__yx * Kt.McorSD__yx, bins_T, rad_scale=1, mode='sum') / \
                Kt.radialProfile(Kt.McorSD__yx, bins_T, rad_scale=1, mode='sum')
    M_T = Kt.radialProfile(Kt.McorSD__yx, bins_T, rad_scale=1)
    atf_Ti = Kt.integrated_at_flux * np.ones_like(binc_T) 
    AV_Ti = Kt.integrated_keywords['A_V'] * np.ones_like(binc_T)
    azm_Ti = Kt.integrated_alogZ_mass * np.ones_like(binc_T)
    M_Ti = Kt.integrated_Mcor / getArea(Kt, radius_HLR) * np.ones_like(binc_T)
    
    bins_B = np.arange(int(Kb.getHLR_pix() * radius_HLR))
    binc_B = bins_B[:-1] + 0.5
    #atf_B = Kb.radialProfile(Kb.at_flux__yx * Kb.LobnSD__yx, bins_B, rad_scale=1, mode='sum') / \
    #            Kb.radialProfile(Kb.LobnSD__yx, bins_B, rad_scale=1, mode='sum') 
    #AV_B = Kb.radialProfile(Kb.A_V__yx, bins_B, rad_scale=1)
    #azm_B = Kb.radialProfile(Kb.alogZ_mass__yx * Kb.McorSD__yx, bins_B, rad_scale=1, mode='sum') / \
    #            Kb.radialProfile(Kb.McorSD__yx, bins_B, rad_scale=1, mode='sum') 
    #M_B = Kb.radialProfile(Kb.McorSD__yx, bins_B, rad_scale=1)
    atf_B = Kb.integrated_at_flux * np.ones_like(binc_B) 
    AV_B = Kb.integrated_keywords['A_V'] * np.ones_like(binc_B)
    azm_B = Kb.integrated_alogZ_mass * np.ones_like(binc_B)
    M_B = Kb.integrated_Mcor / getArea(Kb, radius_HLR) * np.ones_like(binc_B)
    
    bins_D = np.arange(int(Kd.getHLR_pix() * radius_HLR))
    #bins_D = np.arange(30)
    binc_D = bins_D[:-1] + 0.5
    #atf_D = Kd.radialProfile(Kd.at_flux__yx * Kd.LobnSD__yx, bins_D, rad_scale=1, mode='sum') / \
    #            Kd.radialProfile(Kd.LobnSD__yx, bins_D, rad_scale=1, mode='sum') 
    #AV_D = Kd.radialProfile(Kd.A_V__yx, bins_D, rad_scale=1)
    #azm_D = Kd.radialProfile(Kd.alogZ_mass__yx * Kd.McorSD__yx, bins_D, rad_scale=1, mode='sum') / \
    #            Kd.radialProfile(Kd.McorSD__yx, bins_D, rad_scale=1, mode='sum') 
    #M_D = Kd.radialProfile(Kd.McorSD__yx, bins_D, rad_scale=1)
    atf_D = Kd.integrated_at_flux * np.ones_like(binc_D) 
    AV_D = Kd.integrated_keywords['A_V'] * np.ones_like(binc_D)
    azm_D = Kd.integrated_alogZ_mass * np.ones_like(binc_D)
    M_D = Kd.integrated_Mcor / getArea(Kd, radius_HLR) * np.ones_like(binc_D)
    
    fig = plt.figure(2,  figsize=(8, 6))
    gs = plt.GridSpec(2, 2, height_ratios=[1.0, 1.0])
    ax = plt.subplot(gs[0, 0])
    ax.plot(binc_T, atf_T, 'k', label=r'Original')
    ax.plot(binc_T, atf_Ti, 'k--', label=r'Original (int)')
    ax.plot(binc_B, atf_B, 'r--', label=r'Bulge (int)')
    ax.plot(binc_D, atf_D, 'b--', label=r'Disk (int)')
    ax.set_xlabel(r'radius $[arcsec]$')
    ax.set_ylabel(r'$\langle \log\ t_\star \rangle_{flux}\ [yr]$')
    #     ax.legend(loc='lower right')
    
    ax = plt.subplot(gs[0, 1])
    ax.plot(binc_T, AV_T, 'k', label=r'Original')
    ax.plot(binc_T, AV_Ti, 'k--', label=r'Original (int)')
    ax.plot(binc_B, AV_B, 'r--', label=r'Bulge (int)')
    ax.plot(binc_D, AV_D, 'b--', label=r'Disk (int)')
    ax.set_xlabel(r'radius $[arcsec]$')
    ax.set_ylabel(r'$A_V$')
    ax.legend(loc='upper right')
    
    ax = plt.subplot(gs[1, 0])
    ax.plot(binc_T, azm_T, 'k', label=r'Original')
    ax.plot(binc_T, azm_Ti, 'k--', label=r'Original (int)')
    ax.plot(binc_B, azm_B, 'r--', label=r'Bulge (int)')
    ax.plot(binc_D, azm_D, 'b--', label=r'Disk (int)')
    ax.set_xlabel(r'radius $[arcsec]$')
    ax.set_ylabel(r'$\langle \log (Z_\star/Z_\odot) \rangle_{mass}$')
    #     ax.legend(loc='upper right')
    
    ax = plt.subplot(gs[1, 1])
    ax.plot(binc_T, M_T, 'k', label=r'Original')
    ax.plot(binc_T, M_Ti, 'k--', label=r'Original (int)')
    ax.plot(binc_B, M_B, 'r--', label=r'Bulge (int)')
    ax.plot(binc_D, M_D, 'b--', label=r'Disk (int)')
    ax.set_xlabel(r'radius $[arcsec]$')
    ax.set_ylabel(r'$\mu_\star\ [M_\odot / pc^2]$')
    #     ax.legend(loc='upper right')
    
    ax.text(0.95, 0.9, 'Bulge mass: %.1f %%, light: %.1f %%' % (m_B, l_B), color='r', ha='right', transform=ax.transAxes)
    ax.text(0.95, 0.8, 'Disk mass: %.1f %%, light: %.1f %%' % (m_D, l_D), color='b', ha='right', transform=ax.transAxes)
    
    plt.suptitle('%s - %s - Bgsd6e' % (galaxyId, gal['Name']))
    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    pdf.savefig(fig)
    if args.debug:
        plt.show()
################################################################################


parser = argparse.ArgumentParser(description='Plot Bulge/Disk synthesis.')

parser.add_argument('--sample', dest='sample', default= 'data/tables/sample006a',
                    help='Sample file. Default: sample004.')
parser.add_argument('--cube-dir', dest='cubeDir', default= 'data/superfits',
                    help='Sample file. Default: sample004.')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Show plots.')

args = parser.parse_args()
sample = load_sample(args.sample)
sampleId = path.basename(args.sample)
pdf = plot_setup('plots/%s_synthesis.pdf' % (sampleId))
for gal in sample:
    plotall(gal, sampleId, args, pdf)

pdf.close()



