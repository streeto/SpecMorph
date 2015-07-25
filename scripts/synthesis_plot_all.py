
'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
def line(x1, y1, x2, y2):
    x = np.linspace(x1, x2, 100)
    y = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
    return x, y
################################################################################
    

################################################################################
def get_best_SSP_age_met(K, base):
    i_best_SSP = K.integrated_keywords['INDEX_BEST_SSP'] - 1
    age = base.age_base[i_best_SSP]
    Z = base.Z_base[i_best_SSP]
    return np.log10(age), np.log10(Z / 0.019)
################################################################################


################################################################################
def plotall(sample, args):
    sampleId = path.basename(args.sample)
    pdf = plot_setup('plots/%s_synthesis_all.pdf' % (sampleId))
    base = load_base(args.baseFile)
    
    at_flux_Ts = []
    alogZ_mass_Ts = []
    at_flux_T = []
    alogZ_mass_T = []
    best_logt_T = []
    best_logZ_T = []
    at_flux_B = []
    alogZ_mass_B = []
    best_logt_B = []
    best_logZ_B = []
    at_flux_D = []
    alogZ_mass_D = []
    best_logt_D = []
    best_logZ_D = []
    name = {}
    name_best = {}
    
    for gal in sample:
        galaxyId = califa_id_from_cube(gal['cube'])

        try:
            Kt = getCube(args.cubeDir, galaxyId, sampleId, 'total')
            Kb = getCube(args.cubeDir, galaxyId, sampleId, 'bulge', Kt.HLR_pix)
            Kd = getCube(args.cubeDir, galaxyId, sampleId, 'disk', Kt.HLR_pix)
        except:
            print '***** Skipped galaxy %s' % galaxyId
            continue
        
        best_logt, best_logZ = get_best_SSP_age_met(Kt, base)
        best_logt_T.append(best_logt)
        best_logZ_T.append(best_logZ)
        best_logt, best_logZ = get_best_SSP_age_met(Kd, base)
        best_logt_D.append(best_logt)
        best_logZ_D.append(best_logZ)
        best_logt, best_logZ = get_best_SSP_age_met(Kb, base)
        best_logt_B.append(best_logt)
        best_logZ_B.append(best_logZ)
        name_best[(np.asscalar(best_logZ), np.asscalar(best_logt))] = galaxyId
    
        at_flux_Ts.append((Kt.at_flux__z * Kt.Lobn__z).sum() / Kt.Lobn__z.sum())
        alogZ_mass_Ts.append((Kt.alogZ_mass__z * Kt.Mcor__z).sum() / Kt.Mcor__z.sum())
        at_flux_T.append(np.asscalar(Kt.integrated_at_flux))
        alogZ_mass_T.append(np.asscalar(Kt.integrated_alogZ_mass))
        at_flux_B.append(np.asscalar(Kb.integrated_at_flux))
        alogZ_mass_B.append(np.asscalar(Kb.integrated_alogZ_mass))
        at_flux_D.append(np.asscalar(Kd.integrated_at_flux))
        alogZ_mass_D.append(np.asscalar(Kd.integrated_alogZ_mass))
        name[(np.asscalar(Kb.integrated_alogZ_mass), np.asscalar(Kb.integrated_at_flux))] = galaxyId
    
    fig = plt.figure(1,  figsize=(5, 7))
    gs = plt.GridSpec(2, 1, height_ratios=[1.0, 1.0])
    ax = plt.subplot(gs[0])
    ax.scatter(alogZ_mass_B, at_flux_B, color='r', label=r'Bulge')
    ax.scatter(alogZ_mass_D, at_flux_D, color='b', label=r'Disk')
    ax.set_title('Regular fit')
    for zB, tB, zD, tD in zip(alogZ_mass_B, at_flux_B, alogZ_mass_D, at_flux_D):
        ax.text(zB, tB, name[(zB, tB)], color='k', ha='right')
        try:
            x, y = line(zB, tB, zD, tD)
            ax.plot(x, y, 'k:')
        except:
            print 'Can\'t draw line:', zB, tB, zD, tD
    ax.set_xlabel(r'$\langle \log (Z_\star/Z_\odot) \rangle_{mass}$')
    ax.set_ylabel(r'$\langle \log\ t_\star \rangle_{flux}\ [yr]$')
    ax.set_xlim(-0.8, 0.4)
    ax.set_ylim(9.0, 10.5)
    ax.legend(loc='lower left')

    ax = plt.subplot(gs[1])
    ax.scatter(best_logZ_B, best_logt_B, color='r', label=r'Bulge')
    ax.scatter(best_logZ_D, best_logt_D, color='b', label=r'Disk')
    ax.set_title('Best SSP fits')
    for zB, tB, zD, tD in zip(best_logZ_B, best_logt_B, best_logZ_D, best_logt_D):
        ax.text(zB, tB, name_best[(zB, tB)], color='k', ha='right')
        try:
            x, y = line(zB, tB, zD, tD)
            ax.plot(x, y, 'k:')
        except:
            print 'Can\'t draw line:', zB, tB, zD, tD
    ax.set_xlabel(r'$\log (Z_\star/Z_\odot)$')
    ax.set_ylabel(r'$\log\ t_\star\ [yr]$')
    ax.set_xlim(-0.8, 0.4)
    ax.set_ylim(9.0, 10.5)
    #ax.legend(loc='lower left')
    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    
    plt.show()
    
    pdf.savefig(fig)
    pdf.close()

################################################################################


parser = argparse.ArgumentParser(description='Plot Bulge/Disk synthesis.')

parser.add_argument('--sample', dest='sample', default= 'data/tables/sample006a',
                    help='Sample file. Default: sample004.')
parser.add_argument('--cube-dir', dest='cubeDir', default= 'data/superfits',
                    help='Sample file. Default: sample004.')
parser.add_argument('--base-file', dest='baseFile', default= 'data/starlight/BASE.gsd6e',
                    help='Base file. Default: data/starlight/BASE.gsd6e.')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Show plots.')

args = parser.parse_args()
sample = load_sample(args.sample)
plotall(sample, args)




