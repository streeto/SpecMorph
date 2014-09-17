'''
Created on Sep 24, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
from imfit import Imfit, SimpleModelDescription, function_description
from imfit.psf import moffat_psf

import numpy as np
from os import path


def bulge_function(I_e, r_e, n, PA, ell):
    bulge = function_description('Sersic', name='bulge')
    bulge.I_e.setValue(I_e, vmin=1e-33, vmax=10*I_e)
    bulge.r_e.setValue(r_e, vmin=1e-33, vmax=10*r_e)
    bulge.n.setValue(n, vmin=0.5, vmax=5.9)
    bulge.PA.setValue(PA, vmin=0, vmax=180)
    bulge.ell.setValue(ell, vmin=0, vmax=1)
    return bulge


def disk_function(I_0, h, PA, ell):
    disk = function_description('Exponential', name='disk')
    disk.I_0.setValue(I_0, vmin=1e-33, vmax=10*I_0)
    disk.h.setValue(h, vmin=1e-33, vmax=10*h)
    disk.PA.setValue(PA, vmin=0, vmax=180)
    disk.ell.setValue(ell, vmin=0, vmax=1)
    return disk


def galaxy_model(x0, y0, bulge=False, disk=False,
                 PA_b=0.0, ell_b=0.0, I_e=0.0, r_e=0.0, n=0.0, PA_d=0.0, ell_d=0.0, I_0=0.0, h=0.0):
    if not (bulge or disk):
        raise Exception('At least one of bulge or disk must be set.') 
    model = SimpleModelDescription()
    model.x0.setValue(x0, vmin=x0-10, vmax=x0+10)
    model.y0.setValue(y0, vmin=y0-10, vmax=y0+10)
    if bulge:
        bulge = bulge_function(I_e, r_e, n, PA_b, ell_b)
        model.addFunction(bulge)
    if disk:
        disk = disk_function(I_0, h, PA_d, ell_d)
        model.addFunction(disk)
    return model



def create_image(shape, PSF, model):
    imfit = Imfit(model, PSF, quiet=True)
    image = imfit.getModelImage(shape)
    return image


def create_model_images(shape, PSF, model):
    x0 = model.x0.value
    y0 = model.y0.value

    I_e = model.bulge.I_e.value
    r_e = model.bulge.r_e.value
    n = model.bulge.n.value
    PA_b = model.bulge.PA.value
    ell_b = model.bulge.ell.value
    
    I_0 = model.disk.I_0.value
    h = model.disk.h.value
    PA_d = model.disk.PA.value
    ell_d = model.disk.ell.value
    
    bulge_model = galaxy_model(x0, y0, bulge=True, I_e=I_e, r_e=r_e, n=n, PA_b=PA_b, ell_b=ell_b)
    disk_model = galaxy_model(x0, y0, disk=True, I_0=I_0, h=h, PA_d=PA_d, ell_d=ell_d)
    
    bulge = create_image(shape, PSF, bulge_model)
    disk = create_image(shape, PSF, disk_model)
    return bulge, disk
    
    
def get_images(db):
    K = fitsQ3DataCube(db, smooth=False)
    mask = ~K.qMask
    qSignal = np.ma.array(K.qSignal, mask=mask)
    qNoise = np.ma.array(K.qNoise, mask=mask)
    
#     PA, ba = K.getEllipseParams()
    
    model = galaxy_model(K.x0, K.y0, bulge=True, disk=True,
                         I_e=1, r_e=12, n=1.5, PA_b=90.0, ell_b=0.5,
                         I_0=1, h=12, PA_d=90, ell_d=0.5)
    PSF = moffat_psf(fwhm=2.4, size=9)
    fitter = Imfit(model, PSF, quiet=True)
    fitter.fit(qSignal, qNoise)
    model_image = fitter.getModelImage()
    model = fitter.getModelDescription()
    bulge_image, disk_image = create_model_images(qSignal.shape, PSF, model)
    bulge_image = np.ma.array(bulge_image, mask=mask)
    disk_image = np.ma.array(disk_image, mask=mask)
    bulge_frac = bulge_image / model_image
    disk_frac = disk_image / model_image
    bulge_noise = np.sqrt(bulge_frac) * qNoise
    disk_noise = np.sqrt(disk_frac) * qNoise
    return model, qSignal, qNoise, model_image, bulge_image, bulge_noise, disk_image, disk_noise


def get_spectra(basesDir, y, o):
    file_spec_02Gyr = path.join(basesDir, y)
    file_spec_12Gyr = path.join(basesDir, o)
    
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

    return wl[::20], spec_y[::20], spec_o[::20]


def fit_spectra(full_spectra, full_noise, flux_unit, init_model):
    PSF = moffat_psf(fwhm=2.4, size=9)
    fitter = Imfit(init_model, PSF, quiet=True)
    models = []
    for i in xrange(0, full_spectra.shape[-1]):
#         print '#'*50, i
        f__l = full_spectra[..., i] / flux_unit
        err__l = full_noise[..., i] / flux_unit
        fitter.fit(f__l, err__l)
        if fitter.nValidPixels < 1000:
            print '(%d) Too many masked pixels.' % i
        if fitter.nPegged < 0:
            print '(%d) Pegged parameters.' % i
        m = fitter.getModelDescription()
#         print m.bulge.I_e.value, m.bulge.r_e.value, m.disk.I_0.value, m.disk.h.value
        models.append(m)
    return models


# def save_spectra(models, flux_unit, imshape):
#     PSF = moffat_psf(fwhm=2,4, size=9)
#     shape = (len(models),) + imshape 
#     bulge_spec = np.empty(shape)
#     disk_spec = np.empty(shape)
#     for i, m in enumerate(models):
#         print '#'*50, i
#         fitter = Imfit(m, PSF, quiet=False)
#         spec[i] = fitter.getModelImage(imshape)
# #         print m.bulge.I_e.value, m.bulge.r_e.value, m.disk.I_0.value, m.disk.h.value
#         models.append(m)
#     return models


def plot_spec_model(wl, models, orig_model, spec_y, spec_o):
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
    I_e = np.array(I_e) / spec_o
    r_e = np.array(r_e)
    PA_b = np.array(PA_b)
    ell_b = np.array(ell_b)
    I_0 = np.array(I_0) / spec_y
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
    plotpars = {'legend.fontsize': 12,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'text.fontsize': 18,
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
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylabel(r'Relative intensity')
    plt.legend(loc='lower right')

    plt.subplot(222)
    plt.plot(wl, r_e, 'r', label=r'$r_e$')
    plt.plot(wl, h, 'b', label=r'$h$')
    plt.hlines(orig_r_e, xmin=wl.min(), xmax=wl.max(), colors='r', linestyle='dotted')
    plt.hlines(orig_h, xmin=wl.min(), xmax=wl.max(), colors='b', linestyle='dotted')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylabel(r'Radial scale $[arcsec]$')
    plt.legend()

    plt.subplot(223)
    plt.plot(wl, PA_b, 'r', label=r'PA bulge')
    plt.plot(wl, PA_d, 'b', label=r'PA disk')
    plt.hlines(orig_PA_b, xmin=wl.min(), xmax=wl.max(), colors='r', linestyle='dotted')
    plt.hlines(orig_PA_d, xmin=wl.min(), xmax=wl.max(), colors='b', linestyle='dotted')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylabel(r'Position Angle $[degrees]$')
    plt.legend()

    plt.subplot(224)
    plt.plot(wl, ell_b, 'r', label=r'ell bulge')
    plt.plot(wl, ell_d, 'b', label=r'ell disk')
    plt.hlines(orig_ell_b, xmin=wl.min(), xmax=wl.max(), colors='r', linestyle='dotted')
    plt.hlines(orig_ell_d, xmin=wl.min(), xmax=wl.max(), colors='b', linestyle='dotted')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylabel(r'Ellipticity $(1 - b/a)$')
    plt.legend()
    
    plt.savefig('K0846_mock_decomposition.pdf')
    plt.close(1)
    
    
def plot_spectra(wl, f_y, f_o):
    import matplotlib.pyplot as plt
    plotpars = {'legend.fontsize': 12,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'text.fontsize': 18,
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
    f = plt.figure(1,  figsize=(6, 5))
    f.set_tight_layout(True)

    plt.plot(wl, f_o, 'r', label=r'$12\ Gyr$')
    plt.plot(wl, f_y, 'b', label=r'$3\ Gyr$')
    plt.xlabel(r'wavelength $[\AA]$')
    plt.ylabel(r'Relative intensity')
    plt.legend(loc='lower right')
    
    plt.savefig('K0846_spectra.pdf')
    plt.close(1)
################################################################################
    
flux_unit = 1e-16

db = '../../../cubes.200/K0846_synthesis_eBR_v01_q036.d13c512.ps03.k2.mC.CCM.Bgsd61.fits'
basesDir = '../../data/starlight/BasesDir/'
y = 'Mun1.30Zp0.00T02.5119.K'
o = 'Mun1.30Zp0.00T12.5893.K'

wl, spec_y, spec_o = get_spectra(basesDir, y, o)
plot_spectra(wl, spec_y, spec_o)

# Initial fit of qSignal to get the original bulge/disk components.
model, qSignal, qNoise, model_image, bulge_image, bulge_noise, disk_image, disk_noise = get_images(db)

# Bulge with old population, disk with young population. 
bulge_spectra = bulge_image[...,np.newaxis] * flux_unit * spec_o 
disk_spectra = disk_image[...,np.newaxis] * flux_unit * spec_y
full_spectra = bulge_spectra + disk_spectra
# qNoise = np.ones_like(qSignal) * 0.1
# Add gaussian noise to spectra.
for i in xrange(qNoise.shape[0]):
    for j in xrange(qNoise.shape[1]):
        full_spectra[i,j] += np.random.normal(0.0, qNoise[i,j] * flux_unit, len(wl))

full_noise = qNoise[...,np.newaxis] * flux_unit * np.ones(len(wl))

# Fit the individual wavelength-wise slices.
# model.disk.h.fixed = True
# model.disk.PA.fixed = True
# model.disk.ell.fixed = True
# model.bulge.r_e.fixed = True
# model.bulge.PA.fixed = True
# model.bulge.ell.fixed = True
models = fit_spectra(full_spectra, full_noise, flux_unit, model)

# Plot the bulge/disk parameters.
plot_spec_model(wl, models, model, spec_y, spec_o)


