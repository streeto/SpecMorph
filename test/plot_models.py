'''
Created on 10/09/2014

@author: andre
'''

import numpy as np
from pylab import normpdf
import matplotlib.pyplot as plt
import sys
from os import path
from imfit import Imfit, function_description, SimpleModelDescription
from pycasso.util import radialProfile

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


################################################################################
def rad_prof(model, bins):
    imfit = Imfit(model, quiet=False)
    image = imfit.getModelImage((model.y0.value * 2, model.x0.value * 2))
    return radialProfile(image, bins, x0, y0, pa=0.0, ba=1.0)
################################################################################


x0 = 100
y0 = 100
r_e = 50.0

label = {0.5: r'$n=0.5$',
         1: r'$n=1\ (\mathrm{exponencial})$',
         4: r'$n=4\ (\mathrm{de\ Vaucouleurs})$',
         10: r'$n=10$',
         }

line = {0.5: 'k:',
        1: 'b-',
        4: 'r-',
        10: 'k-.',
        }

bins = np.arange(100)
bins_c = bins[1:] - 0.5

sersic = function_description('Sersic')
model = SimpleModelDescription()
model.addFunction(sersic)

model.x0.setValue(x0 + 1)
model.y0.setValue(x0 + 1)
model.Sersic.I_e.setValue(1.0)
model.Sersic.r_e.setValue(r_e)
model.Sersic.PA.setValue(0.0)
model.Sersic.ell.setValue(0.0)
model.Sersic.n.setValue(4)

plot_setup()
width_pt = 448.07378
width_in = width_pt / 72.0 * 0.95
fig = plt.figure(figsize=(width_in, width_in * 0.5))

for n in [0.5, 1, 4, 10]:
    model.Sersic.n.setValue(n)
    radprof = rad_prof(model, bins)
    plt.plot(bins_c / r_e, np.log10(radprof), line[n], label=label[n])

plt.legend(frameon=False)
plt.xlim(0, bins_c.max() / r_e)
plt.ylim(-0.99, 2.99)
plt.xlabel(r'$r/r_e$')
plt.ylabel(r'$\log I(r) / I(r_e)$')

plt.savefig('plots/morphModels.pdf')

plt.show()
