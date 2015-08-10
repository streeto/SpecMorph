# -*- coding: utf-8 -*-
'''
Created on Jun 6, 2013

@author: andre
'''

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import argparse
from os import path
from specmorph.califa.tables import califa_id_from_cube
from matplotlib.patches import Polygon


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
    return pdf
################################################################################


################################################################################
def getCube(root, galaxyId, sampleId, component):
    import atpy
    import pystarlight.io  # @UnusedImport
    filename = 'output_%s/%s_0000_%s.%s.out.bz2' % (component, galaxyId, sampleId, component)
    filepath = path.join(root, filename)
    ts = atpy.TableSet(filepath, type='starlight')
    return ts
################################################################################


################################################################################
def line(x1, y1, x2, y2):
    x = np.linspace(x1, x2, 100)
    y = y1 + (y2 - y1) / (x2 - x1) * (x - x1)
    return x, y
################################################################################
    

################################################################################
def get_best_SSP_age_met(K, base):
    i_best_SSP = K.keywords['index_Best_SSP'] - 1
    age = base.age_base[i_best_SSP]
    Z = base.Z_base[i_best_SSP]
    return np.log10(age), np.log10(Z / 0.019)
################################################################################


################################################################################
def get_average_age_met(ts):
    pop = ts.population
    at_flux = (np.log10(pop.popage_base) * pop.popx).sum() / pop.popx.sum()
    alogZ_mass = (np.log10(pop.popZ_base / 0.019) * pop.popmu_cor).sum() / pop.popmu_cor.sum()
    return at_flux, alogZ_mass
################################################################################


################################################################################
def plotall(sample, args):
    sampleId = path.basename(args.sample)
    pdf = plot_setup('plots/%s_synthesis_all2.pdf' % (sampleId))
    base = load_base(args.baseFile)
    
    at_flux_T = []
    alogZ_mass_T = []
    Mass_T = []
    best_logt_T = []
    best_logZ_T = []
    at_flux_B = []
    alogZ_mass_B = []
    Mass_B = []
    best_logt_B = []
    best_logZ_B = []
    at_flux_D = []
    alogZ_mass_D = []
    Mass_D = []
    best_logt_D = []
    best_logZ_D = []
    name = {}
    name_best = {}
    htype = {}
    htype_best = {}
    
    for gal in sample:
        galaxyId = califa_id_from_cube(gal['cube'])
        if galaxyId not in args.galaxies:
            print '***** Skipped galaxy %s' % galaxyId
            continue

        Kt = getCube(args.cubeDir, galaxyId, sampleId, 'total')
        print gal
        print Kt.population.describe()
        Kb = getCube(args.cubeDir, galaxyId, sampleId, 'bulge')
        Kd = getCube(args.cubeDir, galaxyId, sampleId, 'disk')
        #try:
        #except:
        #    print '***** Skipped galaxy %s' % galaxyId
        #    continue
        best_logt, best_logZ = get_best_SSP_age_met(Kt, base)
        best_logt_T.append(best_logt)
        best_logZ_T.append(best_logZ)
        Mass_T.append(Kt.keywords['Mcor_tot'])
        best_logt, best_logZ = get_best_SSP_age_met(Kd, base)
        best_logt_D.append(best_logt)
        best_logZ_D.append(best_logZ)
        Mass_D.append(Kd.keywords['Mcor_tot'])
        best_logt, best_logZ = get_best_SSP_age_met(Kb, base)
        best_logt_B.append(best_logt)
        best_logZ_B.append(best_logZ)
        Mass_B.append(Kb.keywords['Mcor_tot'])
        name_best[(np.asscalar(best_logZ), np.asscalar(best_logt))] = galaxyId
        htype_best[(np.asscalar(best_logZ), np.asscalar(best_logt))] = gal['hubble_type']

        at_flux, alogZ_mass = get_average_age_met(Kt)
        at_flux_T.append(at_flux)
        alogZ_mass_T.append(alogZ_mass)
        at_flux, alogZ_mass = get_average_age_met(Kd)
        at_flux_D.append(at_flux)
        alogZ_mass_D.append(alogZ_mass)
        at_flux, alogZ_mass = get_average_age_met(Kb)
        at_flux_B.append(at_flux)
        alogZ_mass_B.append(alogZ_mass)
        
        name[(np.asscalar(alogZ_mass), np.asscalar(at_flux))] = galaxyId
        htype[(np.asscalar(alogZ_mass), np.asscalar(at_flux))] = gal['hubble_type']
    
    M_max = np.max(Mass_T)
    s = 100
    width_pt = 448.07378
    width_in = width_pt / 72.0 * 0.95
    fig = plt.figure(figsize=(width_in, width_in * 0.7))
    gs = plt.GridSpec(1, 1, height_ratios=[1.0])
    ax = plt.subplot(gs[0])
    #ax.set_title(u'Média dos vetores de população')
    for zB, tB, zD, tD, zT, tT in zip(alogZ_mass_B, at_flux_B, alogZ_mass_D, at_flux_D, alogZ_mass_T, at_flux_T):
        xy = np.array([[zB, tB], [zD, tD], [zT, tT]])
        xc = xy[:,0].sum() / xy.shape[0]
        yc = xy[:,1].sum() / xy.shape[0]
        ax.text(xc, yc, '%s (%s)' % (name[(zB, tB)], htype[(zB, tB)]), color='k', ha='left', va='bottom')
        p = Polygon(xy, alpha=0.2, color='k')
        ax.add_patch(p)
    ax.scatter(alogZ_mass_B, at_flux_B, s=s * np.array(Mass_B) / M_max, color='r', label=r'bojo')
    ax.scatter(alogZ_mass_D, at_flux_D, s=s * np.array(Mass_D) / M_max, color='b', label=r'disco')
    ax.scatter(alogZ_mass_T, at_flux_T, s=s * np.array(Mass_T) / M_max, color='k', label=r'observado')
    ax.set_xlabel(r'$\langle \log (Z_\star/Z_\odot) \rangle_{\mathrm{Massa}}$')
    ax.set_ylabel(r'$\langle \log\ t_\star \rangle_{\mathrm{Fluxo}}$')
    ax.set_xlim(-0.4, 0.3)
    ax.set_ylim(9.4, 10.0)
    ax.legend(loc='lower left', frameon=False)

#     ax = plt.subplot(gs[1])
#     ax.set_title(u'Ajuste de única SSP')
#     for zB, tB, zD, tD, zT, tT in zip(best_logZ_B, best_logt_B, best_logZ_D, best_logt_D, best_logZ_T, best_logt_T):
#         xy = np.array([[zB, tB], [zD, tD], [zT, tT]])
#         xc = xy[:,0].sum() / xy.shape[0]
#         yc = xy[:,1].sum() / xy.shape[0]
#         ax.text(xc, yc, '%s (%s)' % (name_best[(zB, tB)], htype_best[(zB, tB)]), color='k', ha='left', va='bottom')
#         p = Polygon(xy, alpha=0.2, color='k')
#         ax.add_patch(p)
#     ax.scatter(best_logZ_B, best_logt_B, s=s * np.array(Mass_B) / M_max, color='r', label=r'bojo')
#     ax.scatter(best_logZ_D, best_logt_D, s=s * np.array(Mass_D) / M_max, color='b', label=r'disco')
#     ax.scatter(best_logZ_T, best_logt_T, s=s * np.array(Mass_T) / M_max, color='k', label=r'observado')
#     ax.set_xlabel(r'$\log (Z_\star/Z_\odot)$')
#     ax.set_ylabel(r'$\log\ t_\star$')
#     ax.set_xlim(-0.75, 0.3)
#     ax.set_ylim(9.2, 10.2)
    #ax.legend(loc='lower left')
    gs.tight_layout(fig, rect=[0, 0, 1, 0.97])
    
    plt.show()
    
    pdf.savefig(fig)
    pdf.close()

################################################################################


parser = argparse.ArgumentParser(description='Plot Bulge/Disk synthesis.')

parser.add_argument('galaxies', type=str, nargs='+',
                    help='CALIFA ID. Ex.: K0001 K0002 ...')
parser.add_argument('--sample', dest='sample', default= 'data/tables/sample006a',
                    help='Sample file. Default: sample004.')
parser.add_argument('--cube-dir', dest='cubeDir', default= 'data/integrated',
                    help='Sample file. Default: sample004.')
parser.add_argument('--base-file', dest='baseFile', default= 'data/starlight/BASE.gsd6e',
                    help='Base file. Default: data/starlight/BASE.gsd6e.')
parser.add_argument('--debug', dest='debug', action='store_true',
                    help='Show plots.')

args = parser.parse_args()
sample = load_sample(args.sample)
plotall(sample, args)




