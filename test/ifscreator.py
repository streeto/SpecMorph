'''
Created on 30/05/2014

@author: andre
'''


from specmorph.components import SyntheticSFH
from specmorph.model import BDModel, create_model_images
from specmorph.geometry import distance
from specmorph.util import logger
from specmorph import flags

from pystarlight.util.base import StarlightBase
from imfit import gaussian_psf, convolve_image
import numpy as np
import time
from os import path
import argparse
from tables import openFile
from specmorph.io import save_array
import sys

################################################################################
def parse_args():
    parser = argparse.ArgumentParser(description='Mock Bulge/Disk decomposition.')
    
    parser.add_argument('--model', dest='trueModel',
                        help='File containing the morphological model of the galaxy.')
    parser.add_argument('--base', dest='baseFile', default='data/starlight/BASE.gsd6e',
                        help='File describing the starlight bases.')
    parser.add_argument('--base-dir', dest='baseDir', default='data/starlight/BasesDir',
                        help='Directory containing the base spectra.')
    parser.add_argument('--psf-fwhm', dest='psfFWHM', type=float, default=2.4,
                        help='PSF FWHM (arcseconds).')
    parser.add_argument('--nx', dest='Nx', type=int, default=77,
                        help='X dimension.')
    parser.add_argument('--ny', dest='Ny', type=int, default=72,
                        help='Y dimension.')
    parser.add_argument('--flag-radius', dest='flagRadius', type=int, default=32,
                        help='Flag pixels more distant from the center than this value.')
    parser.add_argument('--flag-badpix', dest='flagThreshold', type=float, default=0.01,
                        help='Fraction of random pixels flagged as bad pixels.')
    parser.add_argument('--noise-fraction', dest='noiseFraction', type=float, default=0.05,
                        help='Noise fraction.')
    parser.add_argument('--flux-unit', dest='fluxUnit', type=float, default=1e-16,
                        help='Flux unit.')
    parser.add_argument('--tau0', dest='tau0', type=float, default=2e9,
                        help='Width of the bulge star formation burst at the nucleus.')
    parser.add_argument('--dtau-dr', dest='dtau_dr', type=float, default=2e8,
                        help='Gradient of tau.')
    parser.add_argument('--db', dest='dbOutput', default='fake_ifs.h5',
                        help='Output HDF5 database path.')
    parser.add_argument('--name', dest='galaxyName', default='default_galaxy',
                        help='Output HDF5 database path.')
    parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                        help='Overwrite data.')
    
    
    return parser.parse_args()
################################################################################


################################################################################
def default_model():
    true_model = BDModel()
    true_model.x0.setValue(36.0)
    true_model.y0.setValue(33.0)
    true_model.bulge.I_e.setValue(2.0)
    true_model.bulge.r_e.setValue(5.0)
    true_model.bulge.n.setValue(2.0)
    true_model.bulge.ell.setValue(0.1)
    true_model.bulge.PA.setValue(45.0)
    true_model.disk.I_0.setValue(1.0)
    true_model.disk.h.setValue(15.0)
    true_model.disk.ell.setValue(0.1)
    true_model.disk.PA.setValue(45.0)
    return true_model
################################################################################


################################################################################
def get_model(model_file=None, with_default=False):
    if model_file is not None:
        try:
            logger.info('Loading model from %s.' % model_file)
            model = BDModel.load(model_file)
            model.wl = 5635.0
            return model
        except:
            raise Exception('Could not read model file %s.' % model_file)
    elif not with_default:
        logger.info('Using default model.')
        return default_model()
    else:
        raise Exception('No model_file and with_default=False, what do you want?')
################################################################################


################################################################################
def tau_r(tau0, dtau_dr, r):
    return tau0 + r * dtau_dr
################################################################################


################################################################################
def get_bulge_distance(shape, model):
    return distance(shape, model.x0.value, model.x0.value,
                    model.bulge.PA.value, model.bulge.ell.value)
################################################################################

        
################################################################################
##########
########## Population model setup
##########
################################################################################

logger.setLevel(-1)
args = parse_args()

logger.info('Loading base %s', path.basename(args.baseFile))
t1 = time.clock()
base = StarlightBase(args.baseFile, args.baseDir)
l_ssp = np.arange(3650.0, 6850.0, 2.0)
f_ssp = base.f_sspResam(l_ssp)
logger.info('Took %.2f seconds to read the base (%d files)' % (time.clock() - t1, base.sspfile.size))
wl_norm_window = (l_ssp < 5680.0) & (l_ssp > 5590.0)

################################################################################
##########
########## Morphology model setup
##########
################################################################################

logger.info('Creating original B-D model.')
norm_model = get_model(args.trueModel, with_default=True)
norm_params = np.array(norm_model.getParams(), dtype=norm_model.dtype)
logger.info('Original model at normalization window:\n%s\n' % str(norm_model))

logger.info('Creating PSF (FWHM = %.2f ")' % args.psfFWHM)
PSF = gaussian_psf(args.psfFWHM, size=15)
# Pad image to avoid artifacts at the borders when convolving.
Ny_psf = PSF.shape[0]
Nx_psf = PSF.shape[1]
imshape_pad = (args.Ny + 2 * Ny_psf, args.Nx + 2 * Nx_psf)
ifsshape_pad = (len(l_ssp),) + imshape_pad
# Fix the model center for padded images.
norm_y0 = norm_model.y0.value + Ny_psf
norm_x0 = norm_model.x0.value + Nx_psf
norm_model.y0.setValue(norm_y0)
norm_model.x0.setValue(norm_x0)

t1 = time.clock()
logger.info('Creating bulge spectra (tau proportional to distance).')
bulge_image_pad, disk_image_pad = create_model_images(norm_model, imshape_pad, PSF=None)

bulge_r = get_bulge_distance(imshape_pad, norm_model)
bulge_ifs_pad = np.empty(ifsshape_pad)
tau_image_pad = np.empty(imshape_pad)
for i in xrange(imshape_pad[0]):
    for j in xrange(imshape_pad[1]):
        r = bulge_r[i, j]
        tau = tau_r(args.tau0, args.dtau_dr, r)
        tau_image_pad[i, j] = tau
        bulge_sfh = SyntheticSFH(base.ageBase)
        bulge_sfh.addExp(base.ageBase.max(), tau, 1.0)
        bulge_spec = (f_ssp * bulge_sfh.massVector()[:, np.newaxis]).sum(axis=1).sum(axis=0)
        bulge_spec /= np.median(bulge_spec[wl_norm_window])
        bulge_ifs_pad[:, i, j] =  bulge_spec * (args.fluxUnit * bulge_image_pad[np.newaxis, i, j])

logger.info('Creating disk spectra.')
disk_sfh = SyntheticSFH(base.ageBase)
disk_sfh.addSquare(0.0, 14e9, 1.0)
disk_spec = (f_ssp * disk_sfh.massVector()[:, np.newaxis]).sum(axis=1).sum(axis=0)
disk_spec /= np.median(disk_spec[wl_norm_window])

logger.info('Creating IFS.')
disk_ifs_pad = disk_spec[..., np.newaxis, np.newaxis] * (args.fluxUnit * disk_image_pad)
full_ifs_pad = bulge_ifs_pad + disk_ifs_pad

logger.info('Convolving IFS with PSF.')
imshape = (args.Ny, args.Nx)
ifsshape = (len(l_ssp),) + imshape
flagged = distance(imshape, norm_x0, norm_y0) > args.flagRadius
bulge_ifs = np.empty(ifsshape)
disk_ifs = np.empty(ifsshape)
for l in xrange(len(l_ssp)):
    bulge_ifs[l] = convolve_image(bulge_ifs_pad[l], PSF)[Ny_psf:-Ny_psf, Nx_psf:-Nx_psf]
    disk_ifs[l] = convolve_image(disk_ifs_pad[l], PSF)[Ny_psf:-Ny_psf, Nx_psf:-Nx_psf]

# Convolution is linear.
full_ifs = bulge_ifs + disk_ifs

# We removed the PSF padding, fix the center of the model again.
norm_y0 = norm_model.y0.value - Ny_psf
norm_x0 = norm_model.x0.value - Nx_psf
norm_model.y0.setValue(norm_y0)
norm_model.x0.setValue(norm_x0)

logger.info('Adding fake gaussian noise to IFS.')
full_ifs_noise = np.sqrt(full_ifs)
full_ifs_noise *= full_ifs_noise.max() / 60.0
full_ifs += np.random.normal(0.0, 1.0, ifsshape) * full_ifs_noise

logger.info('Creating flags.')
flag_ifs = np.zeros(ifsshape, dtype='int')

# Global mask.
r = distance(imshape, imshape[1] / 2, imshape[0] / 2)
nodata = r > args.flagRadius
flag_ifs[:,nodata] |= flags.no_data

# Add bad pixels.
badpix = np.random.random(ifsshape) < args.flagThreshold
flag_ifs[badpix] |= flags.bad_pixel

# TODO: add other flags

logger.info('Took %.2f seconds to create the IFS.' % (time.clock() - t1))

################################################################################
##########
########## Write IFS
##########
################################################################################

logger.info('Saving to storage...')
db = openFile(args.dbOutput, 'a')
if not args.overwrite and args.galaxyName in db.root:
    logger.error('Galaxy %s already exists. Use --overwrite.')
    sys.exit()
try:
    grp = db.getNode('/%s' % args.galaxyName)
except:
    grp = db.createGroup('/', args.galaxyName)

save_array(db, grp, 'bulge_image', bulge_image_pad[Ny_psf:-Ny_psf, Nx_psf:-Nx_psf], args.overwrite)
save_array(db, grp, 'disk_image', disk_image_pad[Ny_psf:-Ny_psf, Nx_psf:-Nx_psf], args.overwrite)
save_array(db, grp, 'wl', l_ssp, args.overwrite)
save_array(db, grp, 'bulge_ifs_nopsf', bulge_ifs_pad[:, Ny_psf:-Ny_psf, Nx_psf:-Nx_psf], args.overwrite)
save_array(db, grp, 'bulge_ifs', bulge_ifs, args.overwrite)
save_array(db, grp, 'disk_ifs_nopsf', disk_ifs_pad[:, Ny_psf:-Ny_psf, Nx_psf:-Nx_psf], args.overwrite)
save_array(db, grp, 'disk_ifs', disk_ifs, args.overwrite)
save_array(db, grp, 'full_ifs', full_ifs, args.overwrite)
save_array(db, grp, 'full_ifs_noise', full_ifs_noise, args.overwrite)
save_array(db, grp, 'flag_ifs', flag_ifs, args.overwrite)

save_array(db, grp, 'psf', PSF, args.overwrite)

save_array(db, grp, 'tau_image', tau_image_pad[Ny_psf:-Ny_psf, Nx_psf:-Nx_psf], args.overwrite)
save_array(db, grp, 'age_base', base.ageBase, args.overwrite)

ifs = db.getNode('/%s/full_ifs' % args.galaxyName)
for k, val in vars(args).iteritems():
    ifs.attrs[k] = val
ifs.attrs['model'] = norm_params

db.close()
logger.info('Storage complete.')

