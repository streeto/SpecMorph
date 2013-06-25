'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import ParametricModel, Parameter, fitting
from astropy.modeling.core import _convert_input, _convert_output

class BulgeModel(ParametricModel):
    param_names = ['I_Be', 'R_e']

    def __init__(self, I_Be, R_e, param_dim=1):
        self._I_Be = Parameter(name='I_Be', val=I_Be, mclass=self, param_dim=param_dim)
        self._R_e = Parameter(name='R_e', val=R_e, mclass=self, param_dim=param_dim, min=R_e/3.0, max=R_e*3.0)
        ParametricModel.__init__(self, self.param_names, n_inputs=1, n_outputs=1, param_dim=param_dim)
        self.linear = False


    def eval(self, r, params):
        return params[0] * np.exp(-7.669 * ((r / params[1])**0.25 - 1))
                                  
                                  
    def deriv(self, params, r, y):
        I_Be, R_e = params
        I_B = self.eval(r, params)
        d_I_Be = I_B / I_Be
        d_Re = I_B * 7.669 / 4.0 / R_e**1.25 * r**0.25
        return np.array([d_I_Be, d_Re]).T 
                                  
                                  
    def __call__(self, r):
        r, fmt = _convert_input(r, self.param_dim)
        result = self.eval(r, self.param_sets)
        return _convert_output(result, fmt)

    
class DiskModel(ParametricModel):
    param_names = ['I_D0', 'R_0']

    def __init__(self, I_D0, R_0, param_dim=1):
        self._I_D0 = Parameter(name='I_D0', val=I_D0, mclass=self, param_dim=param_dim)
        self._R_0 = Parameter(name='R_0', val=R_0, mclass=self, param_dim=param_dim)
        ParametricModel.__init__(self, self.param_names, n_inputs=1, n_outputs=1, param_dim=param_dim)
        self.linear = False

    def eval(self, r, params):
        return params[0] * np.exp(-r / params[1])

    def deriv(self, params, r, y):
        I_D0, R_0 = params
        I_D = self.eval(r, params)
        d_I_D0 = I_D / I_D0
        d_R0 = I_D / R_0**2 * r
        return np.array([d_I_D0, d_R0]).T 
                                  
                                  
    def __call__(self, r):
        r, fmt = _convert_input(r, self.param_dim)
        result = self.eval(r, self.param_sets)
        return _convert_output(result, fmt)



class GalaxyModel(ParametricModel):
    param_names = ['I_Be', 'R_e', 'I_D0', 'R_0']


    def __init__(self, I_Be, R_e, I_D0, R_0, param_dim=1):
        self.linear = False
        self._I_Be = Parameter(name='I_Be', val=I_Be, mclass=self, param_dim=param_dim, min=I_Be/10.0)
        self._R_e = Parameter(name='R_e', val=R_e, mclass=self, param_dim=param_dim, min=R_e/3.0, max=R_e*3.0)
        self._I_D0 = Parameter(name='I_D0', val=I_D0, mclass=self, param_dim=param_dim, min=I_D0/10.0)
        self._R_0 = Parameter(name='R_0', val=R_0, mclass=self, param_dim=param_dim, min=R_0/3.0, max=R_0*3.0)
        ParametricModel.__init__(self, self.param_names, n_inputs=1, n_outputs=1, param_dim=param_dim)
        self.linear = False


    def _disk_eval(self, r, params):
        I_Be, R_e, I_D0, R_0 = params
        return I_D0 * np.exp(-r / R_0)


    def _bulge_eval(self, r, params):
        I_Be, R_e, I_D0, R_0 = params
        return I_Be * np.exp(-7.669 * ((r / R_e)**0.25 - 1))


    def eval(self, r, params):
        _I_B = self._bulge_eval(r, params)
        _I_D = self._disk_eval(r, params)
        return _I_B + _I_D
    

    def deriv(self, params, r, y):
        I_Be, R_e, I_D0, R_0 = params
        I_B = self._bulge_eval(r, params)
        I_D = self._disk_eval(r, params)
        d_I_Be = I_B / I_Be
        d_Re = I_B * (7.669 / 4.0 / R_e**1.25) * r**0.25
        d_I_D0 = I_D / I_D0
        d_R0 = I_D / R_0**2 * r
        return np.array([d_I_Be, d_Re, d_I_D0, d_R0]).T 


    def __call__(self, r):
        r, fmt = _convert_input(r, self.param_dim)
        result = self.eval(r, self.param_sets)
        return _convert_output(result, fmt)





K = fitsQ3DataCube('../../../cubes.200/K0127_synthesis_eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits')
pa, ba = K.getEllipseParams()
K.setGeometry(pa, ba)

Nl = 25
Nr = 40

dl = int(np.floor(K.Nl_obs / Nl))
bins_l = K.l_obs[np.arange(0, K.Nl_obs, dl)]
l = K.l_obs[np.arange(dl/2, K.Nl_obs-dl/2, dl)]

pltPars = {'legend.fontsize': 7, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'font.family': 'Times New Roman', 'text.fontsize': 10}
plt.rcParams.update(pltPars)
plt.figure(1, figsize=(10,10))
plt.clf()
plt.suptitle('K0127 - %s' % K.galaxyName.strip())
gs = plt.GridSpec(5, 5)
bins_r = np.arange(1, Nr+1)

f_norm = 1e-17

params = np.zeros(Nl, dtype=[('I_Be', '<f8'), ('R_e', '<f8'), ('I_D0', '<f8'), ('R_0', '<f8'), ('pa', '<f8'), ('ba', '<f8')])

for i in xrange(Nl):
    print i
    l1 = i*dl
    l2 = (i+1)*dl
#     fl = K.f_syn__lyx[l1:l2].sum(axis=0) / dl
    fl = K.f_syn__lyx[l1]
    pa, ba = K.getEllipseParams(fl)
    K.setGeometry(pa, ba)
    mask = K.qMask & (K.pixelDistance__yx >= 2.0)
    rl = K.pixelDistance__yx[mask]
    sort_ix = np.argsort(rl)
    rl = rl[sort_ix]
    fl = fl[mask][sort_ix] / f_norm

    #print 'I_Be=',fl.max(),' R_e=',K.HLR_pix,' I_D0=',fl.max()/2.0,' R_0=',K.HLR_pix
    galModel = GalaxyModel(I_Be=fl.max(), R_e=K.HLR_pix, I_D0=fl.max()/2.0, R_0=K.HLR_pix)
    galFit = fitting.NonLinearLSQFitter(galModel)
    galFit(rl, fl)
    params[i] = tuple(galModel.parameters) + (pa, ba)
    
    bulgeModel = BulgeModel(galModel._I_Be[0], galModel._R_e[0])
    diskModel = DiskModel(galModel._I_D0[0], galModel._R_0[0])
    print galModel

    ax = plt.subplot(gs[i])
    ax.plot(rl, np.log10(fl), 'b.')
    ax.plot(bins_r, np.log10(galModel(bins_r)), 'r-')
    ax.plot(bins_r, np.log10(bulgeModel(bins_r)), 'r:')
    ax.plot(bins_r, np.log10(diskModel(bins_r)), 'r--')
    ax.set_ylim(-1, 1.5)
    ax.set_title(r'$\lambda = %d \AA$' % l[i], x=0.5, y=0.7, fontsize=10)
    if (i % 5) == 0:
        ax.set_ylabel(r'$\log ( F_\lambda / 10^{-17})$')
    else:
        ax.set_yticklabels([])
    if i >= (25 - 5):
        ax.set_xlabel(r'$r\ [arcsec]$')
    else:
        ax.set_xticklabels([])


plt.rcParams['figure.subplot.wspace'] = 0.25
plt.figure(2, figsize=(10,8))
plt.suptitle('K0127 - %s' % K.galaxyName.strip())
plt.subplot(321)
plt.plot(l, params['I_Be'], 'k-')
plt.xlabel('$\lambda [\AA]$')
plt.ylabel('$I_{Be} / 10^{-17}$')

plt.subplot(322)
plt.plot(l, params['R_e'], 'k-')
plt.xlabel('$\lambda [\AA]$')
plt.ylabel('$R_{e} [arcsec]$')

plt.subplot(323)
plt.plot(l, params['I_D0'], 'k-')
plt.xlabel('$\lambda [\AA]$')
plt.ylabel('$I_{D0} / 10^{-17}$')

plt.subplot(324)
plt.plot(l, params['R_0'], 'k-')
plt.xlabel('$\lambda [\AA]$')
plt.ylabel('$R_{0} [arcsec]$')

plt.subplot(325)
plt.plot(l, params['pa'] * 180 / np.pi, 'k-')
plt.xlabel('$\lambda [\AA]$')
plt.ylabel('$P.A. [rad]$')

plt.subplot(326)
plt.plot(l, params['ba'], 'k-')
plt.xlabel('$\lambda [\AA]$')
plt.ylabel('$b/a$')

