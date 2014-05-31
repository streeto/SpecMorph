'''
Created on 30/05/2014

@author: andre
'''


from specmorph.components import SyntheticSFH
import numpy as np
from pystarlight.util.base import StarlightBase
import matplotlib.pyplot as plt

basedir = '../data/starlight/BasesDir'
basefile = '../data/starlight/BASE.gsd6e'
base = StarlightBase(basefile, basedir)

model = SyntheticSFH(base.ageBase)
t1 = 5e9
tau1 = 1e9
frac1 = 0.1
t2 = 14e9
tau2 = 0.5e9
frac2 = 1.0
model.addExp(t1, tau1, frac1)
model.addExp(t2, tau2, frac2)

sfh = model.sfh()
plt.ioff()
plt.plot(base.ageBase, sfh)
plt.show()

flux = np.zeros_like(base.f_ssp[0])
print sfh, sfh.sum()
flux = (base.f_ssp * sfh[:, np.newaxis, np.newaxis]).sum(axis=1).sum(axis=0)
plt.ioff()
plt.plot(base.l_ssp, flux)
plt.show()

