# -*- coding: utf-8 -*-
'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
import matplotlib.pyplot as plt
from pycasso import fitsQ3DataCube

################################################################################
def plot_setup():
    plotpars = {'legend.fontsize': 12,
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


def fix_k(K):
    from specmorph.kinematics import fix_kinematics
    
    flux = K.f_obs / K.flux_unit
    error = K.f_err / K.flux_unit
    flags = K.f_flag > 0.0
    flags |= flux <= 0.0
    flags |= error <= 0.0
    error[error <= 0.0] = error.max()
    target_vd = np.percentile(K.v_d, 95)
    c = 2.997925e5
    lambda_zero = 5500.0
    dl = lambda_zero * target_vd / c
    print 'Fixing kinematics: v_d = %.1f km/s (%.1f \\AA @ 5500 \\AA) ...' % (target_vd, dl)
    flux, error, flags = fix_kinematics(K.l_obs, flux, error,
                                        flags, K.v_0, K.v_d, target_vd)
    assert np.isfinite(flux[~flags]).all()
    assert np.isfinite(error[~flags]).all()
    assert np.isfinite(flags).all()

    assert (flux[~flags] > 0.0).all()
    assert (error[~flags] > 0.0).all()
    
    return flux * K.flux_unit, error * K.flux_unit, flags, target_vd


K = fitsQ3DataCube('../cubes.px1/K0858_synthesis_eBR_px1_q050.d15a500.ps03.k1.mE.CCM.Bgsd6e.fits')
flux_fix, _, flags_fix, target_vd = fix_k(K)

flux = np.ma.array(K.f_obs, mask=K.f_flag)
flux_fix = np.ma.array(flux_fix, mask=flags_fix)


plot_setup()
width_pt = 448.07378
width_in = width_pt / 72.0 * 0.9
fig = plt.figure(figsize=(width_in, width_in * 0.7))

z = 50
norm = flux[:, z].max()

label = r'$v_0 = %d\,\mathrm{km}/\mathrm{s},\ v_d = %d\,\mathrm{km}/\mathrm{s}$' % (K.v_0[z], K.v_d[z])
plt.plot(K.l_obs, flux[:, z] / norm - 0.1, 'r-', label=label)
plt.plot(K.l_obs, flux_fix[:, z] / norm - 0.1, 'r-', alpha=0.5)

z = 300
norm = flux[:, z].max()

label = r'$v_0 = %d\,\mathrm{km}/\mathrm{s},\ v_d = %d\,\mathrm{km}/\mathrm{s}$' % (K.v_0[z], K.v_d[z])
plt.plot(K.l_obs, flux[:, z] / norm, 'b-', label=label)
plt.plot(K.l_obs, flux_fix[:, z] / norm, 'b-', alpha=0.5)


plt.legend(loc='lower right', frameon=False)
plt.xlim(K.l_obs.min(), K.l_obs.max())
plt.xlim(5100, 5300)

plt.ylim(0.5, 0.9)
plt.xlabel(r'Comprimento de onda $[\mathrm{\AA}]$')
plt.ylabel(u'Fluxo (arbitrário)')

plt.title(u'Correção cinemática - $v_{d,\mathrm{alvo}} = %d\,\mathrm{km}/\mathrm{s}$' % target_vd)

plt.savefig('plots/cinematica.pdf')

plt.show()
