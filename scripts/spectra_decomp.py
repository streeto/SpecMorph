'''
Created on Jun 6, 2013

@author: andre
'''

from specmorph.util import logger, find_nearest_index
from specmorph.califa import CALIFADecomposer, save_qbick_images, califa_id_from_cube
from specmorph.model import bd_initial_model, smooth_models
from specmorph.io import DecompContainer

import numpy as np
from os import path
import argparse
import time


################################################################################
def load_line_mask(line_file, wl):
    import atpy
    import pystarlight.io  # @UnusedImport
    t = atpy.Table(line_file, type='starlight_mask')
    masked_wl = np.zeros(wl.shape, dtype='bool')
    for i in xrange(len(t)):
        l_low, l_upp, line_w, line_name = t[i]
        if line_w > 0.0: continue
        logger.debug('Masking region: %s' % line_name)
        masked_wl |= (wl > l_low) & (wl < l_upp)
    return masked_wl
################################################################################


################################################################################
def load_sample(fname):
    import atpy
    from asciitable import CommentedHeaderReader
    return atpy.Table(fname, type='ascii', Reader=CommentedHeaderReader)
################################################################################


################################################################################
def decomp(cube, sampleId, args):
    galaxyId = califa_id_from_cube(cube)
    logger.info('Starting fit for %s...' % galaxyId)
    dec = CALIFADecomposer(cube, grating='none', nproc=args.nproc)
    dec.setSynthPSF(FWHM=args.psfFWHM, beta=args.psfBeta, size=args.psfSize)
    
    logger.warn('Computing initial model using DE algorithm (takes a LOT of time).')
    t1 = time.time()
    if not path.exists(args.maskFile):
        logger.error('Mask file %s not found.' % args.maskFile)
        exit(1)
    logger.info('Using mask file %s.' % args.maskFile)
    masked_wl = load_line_mask(args.maskFile, dec.wl)
    
    l1 = find_nearest_index(dec.wl, 4500.0)
    l2 = dec.Nl_obs
    cache_file = cube + '.initmodel'
    if not path.exists(cache_file):
        logger.info('Creating gray image for initial model.')
        gray_image, gray_noise, _ = dec.getSpectraSlice(l1, l2, masked_wl)
    else:
        gray_image = None
        gray_noise = None
    initial_model = bd_initial_model(gray_image, gray_noise, dec.PSF, quiet=False, nproc=args.nproc,
                                            cache_model_file=cache_file)
    logger.debug('Refined initial model:\n%s\n' % initial_model)
    logger.warn('Initial model time: %.2f\n' % (time.time() - t1))
    
    t1 = time.time()
    c = DecompContainer()
    c.zones = np.ma.array(dec.K.qZones, mask=dec.K.qZones < 0)
    c.initialParams = initial_model.getParams()
    c.attrs = dict(PSF_FWHM=args.psfFWHM,
                   PSF_beta=args.psfBeta,
                   PSF_size=args.psfSize,
                   box_step=args.boxStep,
                   box_radius=args.boxRadius,
                   orig_file=cube,
                   mask_file=args.maskFile, 
                   object_name=dec.K.galaxyName,
                   flux_unit=dec.flux_unit,
                   distance_Mpc=dec.K.distance_Mpc,
                   x0=dec.K.x0,
                   y0=dec.K.y0,
                   target_vd=dec.targetVd,
                   wl_FWHM=dec.wlFWHM)
    
    models = dec.fitSpectra(step=50*args.boxStep, box_radius=25*args.boxStep,
                            initial_model=initial_model, mode='NM', masked_wl=masked_wl)
    c.firstPassParams = np.array([m.getParams() for m in models], dtype=models[0].dtype)
    logger.info('Done first pass modeling, time: %.2f' % (time.time() - t1))
    
    t1 = time.time()
    logger.info('Smoothing parameters.')
    models = smooth_models(models, dec.wl, degree=1)
    
    logger.info('Starting second pass modeling...')
    models = dec.fitSpectra(step=args.boxStep, box_radius=args.boxRadius,
                            initial_model=models, mode='LM', insist=True, masked_wl=masked_wl)
    logger.info('Done second pass modeling, time: %.2f' % (time.time() - t1))
    
    t1 = time.time()
    logger.info('Computing model spectra...')
    c.total.f_obs = dec.flux[::args.boxStep]
    c.total.f_err = dec.error[::args.boxStep]
    c.total.f_flag = dec.flags[::args.boxStep]
    c.total.mask = dec.K.qMask
    c.total.wl = dec.wl[::args.boxStep]
    
    c.bulge.f_obs, c.disk.f_obs = dec.getModelSpectra(models, args.nproc)
    c.bulge.mask = dec.K.qMask
    c.bulge.wl = dec.wl[::args.boxStep]
    c.disk.mask = dec.K.qMask
    c.disk.wl = dec.wl[::args.boxStep]

    # TODO: better array and dtype handling.
    c.fitParams = np.array([m.getParams() for m in models], dtype=models[0].dtype)
    
    flag_bad_fit = c.fitParams['flag'][:, np.newaxis, np.newaxis] > 0.0
    c.updateErrorsFlags(flag_bad_fit)
    c.updateIntegratedSpec()
    
    logger.info('Saving qbick planes...')
    fname = path.join(args.zoneFileDir, '%s_%s-planes.fits' % (galaxyId, sampleId))
    save_qbick_images(c.total, dec, fname, overwrite=args.overwrite)
    fname = path.join(args.zoneFileDir, '%s_%s-bulge-planes.fits' % (galaxyId, sampleId))
    save_qbick_images(c.bulge, dec, fname, overwrite=args.overwrite)
    fname = path.join(args.zoneFileDir, '%s_%s-disk-planes.fits' % (galaxyId, sampleId))
    save_qbick_images(c.disk, dec, fname, overwrite=args.overwrite)
    
    logger.info('Saving to storage...')
    c.writeHDF5(args.db, sampleId, galaxyId, args.overwrite)
    logger.info('Storage complete, time: %.2f' % (time.time() - t1))
    
    return c
################################################################################

    
################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='Perform Bulge/Disk decomposition.')
    
    parser.add_argument('--sample', dest='sample', default='data/tables/sample004',
                        help='Sample table.')
    parser.add_argument('--db', dest='db', default='data/decomposition.005.h5',
                        help='Output HDF5 database path.')
    
    parser.add_argument('--zone-file-dir', dest='zoneFileDir', default='data/planes',
                        help='Output QBICK-like multiplane FITS image directory.')
    parser.add_argument('--mask-file', dest='maskFile', default='data/starlight/Mask.mE',
                        help='Masked wavelengths while performing first pass fit.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Enable verbose output.')
    
    parser.add_argument('--box-radius', dest='boxRadius', type=int, default=0,
                        help='Spectral running average box radius.')
    parser.add_argument('--box-step', dest='boxStep', type=int, default=1,
                        help='Spectral running average box step.')
    parser.add_argument('--vd', dest='vd', type=float, default=None,
                        help='Target v_d in km/s.')
    parser.add_argument('--psf-fwhm', dest='psfFWHM', type=float, default=2.9,
                        help='PSF FWHM in arcseconds.')
    parser.add_argument('--psf-beta', dest='psfBeta', type=float, default=4.0,
                        help='PSF beta parameter for Moffat profile. If not set, use Gaussian.')
    parser.add_argument('--psf-size', dest='psfSize', type=int, default=15,
                        help='PSF size, in pixels. Must be an odd number.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite data.')
    parser.add_argument('--nproc', dest='nproc', type=int, default=None,
                        help='Number of processors to use.')
    
    return parser.parse_args()
################################################################################


################################################################################
if __name__ =='__main__':
    args = parse_args()
    if args.verbose:
        logger.setLevel(-1)
        logger.debug('Verbose output enabled.')

    sample = load_sample(args.sample)
    sampleId = path.basename(args.sample)

    for gal in sample:
        cube = gal['cube']
        c = decomp(cube, sampleId, args)



