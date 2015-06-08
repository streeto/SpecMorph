'''
Created on 02/06/2015

@author: andre
'''

from os import path
import wx
import argparse

import numpy as np
from pycasso import fitsQ3DataCube
from imfit.psf import moffat_psf

from specmorph.initial_model.initial_model_gui import InitialModelFrame
from specmorph.util import logger

################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='Interactively find initial Bulge/Disk model.')
    
    parser.add_argument('cube', type=str, nargs=1,
                        help='CALIFA synthesis cube (FITS file).')
    parser.add_argument('--model-file', dest='modelFile', default=None,
                        help='Initial model file. Default: cube file with ".initmodel" appended.')
    
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Enable verbose output.')
    
    parser.add_argument('--grating', dest='grating', type=str, default='none',
                        help='Which grating was used (changes the error covariance). V500, V1200 or none.')
    parser.add_argument('--estimate-var', dest='estVar', action='store_true',
                        help='Estimate errors when summing spectra.')

    parser.add_argument('--psf-fwhm', dest='psfFWHM', type=float, default=2.9,
                        help='PSF FWHM in arcseconds.')
    parser.add_argument('--psf-beta', dest='psfBeta', type=float, default=4.0,
                        help='PSF beta parameter for Moffat profile. If not set, use Gaussian.')
    parser.add_argument('--psf-size', dest='psfSize', type=int, default=15,
                        help='PSF size, in pixels. Must be an odd number.')

    parser.add_argument('--nproc', dest='nproc', type=int, default=None,
                        help='Number of processors to use.')
    
    return parser.parse_args()
################################################################################


################################################################################
def run_app(image, noise, psf, model_file, plot_title):
    app = wx.PySimpleApp()
    frame = InitialModelFrame(None, wx.ID_ANY, image, noise, psf, model_file,
                              'Initial Model Finder', plot_title)
    frame.Show()
    app.MainLoop()
################################################################################


args = parse_args()
if args.verbose:
    logger.setLevel(-1)
    logger.debug('Verbose output enabled.')

cube = args.cube[0]
model_file = args.cube[0] + '.initmodel'
title = path.basename(cube)

logger.info('Loading cube: %s' % cube)
K = fitsQ3DataCube(args.cube[0])

logger.info('Creating PSF (FWHM=%f, beta=%f, size=%d)' % (args.psfFWHM, args.psfBeta, args.psfSize))
psf = moffat_psf(args.psfFWHM, args.psfBeta, size=args.psfSize)

flags = ~K.qMask | (K.qSignal <= 0.0) | (K.qNoise <= 0.0)
flux = np.ma.array(K.qSignal, mask=flags)
noise = np.ma.array(K.qNoise, mask=flags)

logger.info('Running GUI...')
run_app(flux, noise, psf, model_file, title)
logger.info('Exiting GUI.')

