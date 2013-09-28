'''
Created on Sep 24, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
from imfit import Imfit, SimpleModelDescription, function_description
from imfit.psf import moffat_psf

import numpy as np
import sys


def bulge_function(I_e, r_e, n, PA, ell):
    bulge = function_description('Sersic', name='bulge')
    bulge.I_e.setValue(I_e, [1e-33, 10*I_e])
    bulge.r_e.setValue(r_e, [1e-33, 10*r_e])
    bulge.n.setValue(n, fixed=True)
    bulge.PA.setValue(PA, [0, 180])
    bulge.ell.setValue(ell, [0, 1])
    return bulge


def disk_function(I_0, h, PA, ell):
    disk = function_description('Exponential', name='disk')
    disk.I_0.setValue(I_0, [1e-33, 10*I_0])
    disk.h.setValue(h, [1e-33, 10*h])
    disk.PA.setValue(PA, [0, 180])
    disk.ell.setValue(ell, [0, 1])
    return disk


def galaxy_model(x0, y0, PA, ell, I_e=None, r_e=None, I_0=None, h=None):
    model = SimpleModelDescription()
    model.x0.setValue(x0, [x0-10, x0+10])
    model.y0.setValue(y0, [y0-10, y0+10])
    if I_e is not None and r_e is not None:
        bulge = bulge_function(I_e, r_e, 4, PA, ell)
        model.addFunction(bulge)
    if I_0 is not None and h is not None:
        disk = disk_function(I_0, h, PA, ell)
        model.addFunction(disk)
    return model



def create_image(shape, PSF, model):
    imfit = Imfit(model, PSF)
    image = imfit.getModelImage(shape)
    return image


def create_model_images(shape, PSF, model):
    x0 = model.x0.value
    y0 = model.y0.value

    I_e = model.bulge.I_e.value
    r_e = model.bulge.r_e.value
    PA_b = model.bulge.PA.value
    ell_b = model.bulge.ell.value
    
    I_0 = model.disk.I_0.value
    h = model.disk.h.value
    PA_d = model.disk.PA.value
    ell_d = model.disk.ell.value
    
    bulge_model = galaxy_model(x0, y0, PA_b, ell_b, I_e=I_e, r_e=r_e, I_0=None, h=None)
    disk_model = galaxy_model(x0, y0, PA_d, ell_d, I_e=None, r_e=None, I_0=I_0, h=h)
    
    bulge = create_image(shape, PSF, bulge_model)
    disk = create_image(shape, PSF, disk_model)
    return bulge, disk
    
    
def get_images():
    db = '../../cubes.200/K0846_synthesis_eBR_v20_q036.d13c512.ps03.k1.mC.CCM.Bgsd01.fits'
    K = fitsQ3DataCube(db, smooth=True)
    mask = ~K.qMask
    qSignal = np.ma.array(K.qSignal, mask=mask)
    qNoise = np.ma.array(K.qNoise, mask=mask)
    
#     PA, ba = K.getEllipseParams()
    
    model = galaxy_model(K.x0, K.y0, PA=90.0, ell=0.5, I_e=1, r_e=25, I_0=1, h=25)
    PSF = moffat_psf(fwhm=3.6, size=51)
    fitter = Imfit(model, PSF)
    fitter.fit(qSignal, qNoise, quiet=True)
    model_image = fitter.getModelImage()
    print fitter.getRawParameters()
    print fitter.getModelDescription()
    bulge_image, disk_image = create_model_images(qSignal.shape, PSF, model)
    bulge_image = np.ma.array(bulge_image, mask=mask)
    disk_image = np.ma.array(disk_image, mask=mask)
    bulge_frac = bulge_image / model_image
    disk_frac = disk_image / model_image
    bulge_noise = np.sqrt(bulge_frac) * qNoise
    disk_noise = np.sqrt(disk_frac) * qNoise
    return model, qSignal, qNoise, model_image, bulge_image, bulge_noise, disk_image, disk_noise


def get_spectra():
    file_spec_02Gyr = '../data/starlight/BasesDir/Mun1.30Zp0.00T02.5119.K'
    file_spec_12Gyr = '../data/starlight/BasesDir/Mun1.30Zp0.00T12.5893.K'
    
    

    spec_y = np.genfromtxt(file_spec_02Gyr, names=['wl', 'flux'])
    spec_o = np.genfromtxt(file_spec_12Gyr, names=['wl', 'flux'])
    wl = spec_y['wl'] # same as spec_o
    spec_y = spec_y['flux']
    spec_o = spec_o['flux']
    
    # Normalize in the 5635\AA window.
    mask = (wl < 5680.0) & (wl > 5590.0)
    y_norm = np.median(spec_y[mask])
    o_norm = np.median(spec_o[mask])
    spec_y /= y_norm
    spec_o /= o_norm

    return wl, spec_y, spec_o


def fit_spectra(full_spectra, full_noise, flux_unit, init_model):
    PSF = moffat_psf(fwhm=3.6, size=51)
    fitter = Imfit(init_model, PSF)
    models = []
    for i in xrange(0, full_spectra.shape[-1], 20):
        print '#'*50, i
        f__l = full_spectra[..., i] / flux_unit
        err__l = full_noise[..., i] / flux_unit
        fitter.fit(f__l, err__l, quiet=True)
        m = fitter.getModelDescription()
#         print m.bulge.I_e.value, m.bulge.r_e.value, m.disk.I_0.value, m.disk.h.value
        models.append(m)
    return models


def plot_spec_model(wl, models, orig_model):
    I_e = []
    r_e = []
    PA_b = []
    ell_b = []
    I_0 = []
    h = []
    PA_d = []
    ell_d = []
    for m in models:
        I_e.append(m.bulge.I_e.value)
        r_e.append(m.bulge.r_e.value)
        PA_b.append(m.bulge.PA.value)
        ell_b.append(m.bulge.ell.value)
        I_0.append(m.disk.I_0.value)
        h.append(m.disk.h.value)
        PA_d.append(m.disk.PA.value)
        ell_d.append(m.disk.ell.value)
    I_e = np.array(I_e)
    r_e = np.array(r_e)
    PA_b = np.array(PA_b)
    ell_b = np.array(ell_b)
    I_0 = np.array(I_0)
    h = np.array(h)
    PA_d = np.array(PA_d)
    ell_d = np.array(ell_d)
    
    orig_I_e = orig_model.bulge.I_e.value
    orig_r_e = orig_model.bulge.r_e.value
    orig_PA_b = orig_model.bulge.PA.value
    orig_ell_b = orig_model.bulge.ell.value
    orig_I_0 = orig_model.disk.I_0.value
    orig_h = orig_model.disk.h.value
    orig_PA_d = orig_model.disk.PA.value
    orig_ell_d = orig_model.disk.ell.value
    
    
    import matplotlib.pyplot as plt
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
#     plt.ioff()
    f = plt.figure(1,  figsize=(8, 6))
    f.set_tight_layout(True)

    plt.subplot(221)
    plt.plot(wl, I_e, 'r', label=r'$I_e$')
    plt.plot(wl, I_0, 'b', label=r'$I_0$')
    plt.hlines(orig_I_e, xmin=wl.min(), xmax=wl.max(), colors='r', linestyle='dotted')
    plt.hlines(orig_I_0, xmin=wl.min(), xmax=wl.max(), colors='b', linestyle='dotted')
    plt.legend()

    plt.subplot(222)
    plt.plot(wl, r_e, 'r', label=r'$r_e$')
    plt.plot(wl, h, 'b', label=r'$h$')
    plt.hlines(orig_r_e, xmin=wl.min(), xmax=wl.max(), colors='r', linestyle='dotted')
    plt.hlines(orig_h, xmin=wl.min(), xmax=wl.max(), colors='b', linestyle='dotted')
    plt.legend()

    plt.subplot(223)
    plt.plot(wl, PA_b, 'r', label=r'PA bulge')
    plt.plot(wl, PA_d, 'b', label=r'PA disk')
    plt.hlines(orig_PA_b, xmin=wl.min(), xmax=wl.max(), colors='r', linestyle='dotted')
    plt.hlines(orig_PA_d, xmin=wl.min(), xmax=wl.max(), colors='b', linestyle='dotted')
    plt.legend()

    plt.subplot(224)
    plt.plot(wl, ell_b, 'r', label=r'ell bulge')
    plt.plot(wl, ell_d, 'b', label=r'ell disk')
    plt.hlines(orig_ell_b, xmin=wl.min(), xmax=wl.max(), colors='r', linestyle='dotted')
    plt.hlines(orig_ell_d, xmin=wl.min(), xmax=wl.max(), colors='b', linestyle='dotted')
    plt.legend()
    
    plt.savefig('K0846_mock_decomposition.png')
    plt.close(1)
    
################################################################################
    
flux_unit = 1e-16

wl, spec_y, spec_o = get_spectra()
# import matplotlib.pyplot as plt
# plt.ioff()
# plt.figure(6)
# plt.plot(wl[::20],spec_o[::20]/spec_o[::20].max(), label='O')
# plt.plot(wl[::20],spec_y[::20]/spec_y[::20].max(), label='Y')
# plt.legend()

# Initial fit of qSignal to get the original bulge/disk components.
model, qSignal, qNoise, model_image, bulge_image, bulge_noise, disk_image, disk_noise = get_images()

# Bulge with old population, disk with young population. 
bulge_spectra = bulge_image[...,np.newaxis] * flux_unit * spec_o 
disk_spectra = disk_image[...,np.newaxis] * flux_unit * spec_y
full_spectra = bulge_spectra + disk_spectra
# Add gaussian noise to spectra.
for i in xrange(qNoise.shape[0]):
    for j in xrange(qNoise.shape[1]):
        full_spectra[i,j] += np.random.normal(0.0, qNoise[i,j] * flux_unit, len(wl))

full_noise = qNoise[...,np.newaxis] * flux_unit * np.ones(len(wl))

# Fit the individual wavelength-wise slices.
models = fit_spectra(full_spectra, full_noise, flux_unit, model)

# Plot the bulge/disk parameters.
plot_spec_model(wl[::20], models, model)


