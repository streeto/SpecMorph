'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
import numpy as np
import matplotlib.pyplot as plt

K = fitsQ3DataCube('../../cubes.200/K0127_synthesis_eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits')
pa, ba = K.getEllipseParams()
K.setGeometry(pa, ba)

Nl = 25
Nr = 30

dl = int(np.floor(K.Nl_obs / Nl))
bins_l = K.l_obs[np.arange(0, K.Nl_obs, dl)]
l = K.l_obs[np.arange(dl/2, K.Nl_obs-dl/2, dl)]

pa__l = np.empty(Nl)
ba__l = np.empty(Nl)
radprof__lr = np.empty((Nl, Nr))
plt.figure(1)
plt.clf()
gs = plt.GridSpec(5, 5)
bins_r = np.arange(0, Nr+1)
for i in xrange(Nl):
    print i
    l1 = i*dl
    l2 = (i+1)*dl
    fl = K.f_syn__lyx[l1:l2].sum(axis=0) / dl
    pa__l[i], ba__l[i] = K.getEllipseParams(fl)
    K.setGeometry(pa__l[i], ba__l[i])
    fl = fl[K.qMask]
    rl = K.pixelDistance__yx[K.qMask]
    ax = plt.subplot(gs[i])
    ax.plot(rl, np.log10(fl), '.')
    ax.set_ylim(-18, -15.5)
    ax.set_title(r'$\lambda = %d \AA$' % l[i])
    ax.set_xlabel(r'$r\ [arcsec]$')
    ax.set_ylabel(r'$\log F_\lambda [erg / s / cm^2 / \AA]$')
#     fl, mask = K.fillImage(fl, mode='convex')
#     radprof__lr[i] = K.radialProfile(fl, bins_r, rad_scale=1, mask=mask)

# plt.pcolormesh(bins_r, bins_l, np.log10(radprof__lr))