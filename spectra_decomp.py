'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso.util import logger
from specmorph import BulgeDiskDecomposition
from specmorph.model import GalaxyModel
from specmorph.qbick import integrated_spec, flag_big_error, flag_small_error, calc_sn
import pystarlight.io  # @UnusedImport

from tables import openFile, Filters
import numpy as np

from os import path
import argparse
import time
from tables.atom import Atom
import pyfits
import atpy


################################################################################
def get_planes_image(l_obs, f__lz, l_mask, decomp):
    f_obs = np.ma.masked_invalid(f__lz['f_obs'][l_mask], copy=True)
    f_obs.fill_value = 0.0
    f_obs = f_obs.filled()

    f_flag = f__lz['f_flag'][l_mask]
    l = l_obs[l_mask]
    # FIXME: Dezonification?
    f_obs__lyx = decomp.zoneToYX(f_obs, extensive=True, surface_density=False).filled()
    f_flag__lyx = decomp.zoneToYX(f_flag, extensive=False).filled()
    
    planes = np.zeros(shape=(decomp.N_y, decomp.N_x),
                      dtype=[('Signal', 'float64'), ('Noise', 'float64'),
                             ('Sn', 'float64'), ('ZonesNoise', 'float64'),
                             ('ZonesSn', 'float64')])

    _, qZoneNoise__z, qZoneSn__z = calc_sn(l, f_obs, f_flag)
    planes['ZonesNoise'] = decomp.zoneToYX(qZoneNoise__z, extensive=False, fill_value=0.0).filled()
    planes['ZonesSn'] = decomp.zoneToYX(qZoneSn__z, extensive=False, fill_value=0.0).filled()
    
    mask = decomp.qMask
    snout = calc_sn(l, f_obs__lyx[:,mask], f_flag__lyx[:,mask])
    planes['Signal'][mask] = snout[0]
    planes['Noise'][mask] = snout[1]
    planes['Sn'][mask] = snout[2]
    return  planes
################################################################################


################################################################################
def save_qbick_planes(planes, K, filename):
    # Update planes
    phdu = K.getPrimaryHdu()
    for pname in planes.dtype.names:
        planeId = K._planeIndex[pname]
        phdu.data[planeId] = planes[pname]
        
    # TODO: remove all pycasso headers
    for key in phdu.header.keys():
        if key.startswith('SYN'):
            phdu.header.remove(key)
    hdulist = pyfits.HDUList()
    hdulist.append(phdu)
    hdulist.writeto(filename, clobber=True)
################################################################################


################################################################################
def save_compound_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        print 'Removing existing group %s' % name
        parent._f_getChild(name)._f_remove(recursive=True)
    grp = db.createGroup(parent, name)
    # HACK: pytables does not support compound dtypes.
    for field in data.dtype.names:
        save_array(db, grp, field, data[field], overwrite)
################################################################################

    
################################################################################
def save_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        print 'Removing existing array %s' % name
        parent._f_getChild(name)._f_remove()
    ca = db.createCArray(parent, name, Atom.from_dtype(data.dtype), data.shape, filters=Filters(1, 'blosc'))
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled()
    ca[...] = data
################################################################################


################################################################################
def smooth_param_median(param, flags):
    flag_ok = (flags == 0)
    average_p = np.median(param[flag_ok])
    return np.ones_like(param) * average_p
################################################################################


################################################################################
def smooth_param_box(param, flags, radius):
    p = param.copy()
    N_par = len(param)
    for i in xrange(N_par):
        l1 = max(0, i - radius)
        l2 = min(i + radius, N_par - 1)
        flag_ok = (flags[l1:l2] == 0)
        if flag_ok.any():
            p[i] = np.median(param[l1:l2][flag_ok])
    return p
################################################################################


################################################################################
def smooth_param_polynomial(param, wl, flags, l_obs, degree=1):
    flag_ok = (flags == 0) & (wl > 4500.0)
    from astropy.modeling import models, fitting
    line = models.Polynomial1D(degree)
    fit = fitting.LinearLSQFitter()
    param_fitted = fit(line, wl[flag_ok], param[flag_ok])
    return param_fitted(l_obs)
################################################################################


################################################################################
def smooth_models(models, wl):
    params = np.array([m.getParams() for m in models], dtype=models[0].dtype)
    smooth_params = np.empty(len(wl), dtype=params[0].dtype)    
    param_wl = params['wl']
    param_flag = params['flag']

    for p in params.dtype.names:
        if p in ['wl', 'flag', 'chi2', 'n_pix']: continue
        smooth_params[p] = smooth_param_polynomial(params[p], param_wl, param_flag, wl, degree=1)
    
    models = []
    for i in xrange(len(smooth_params)):
        m = GalaxyModel.fromParamVector(smooth_params[i])
        m.x0.fixed=True
        m.y0.fixed=True
        m.bulge.r_e.fixed=True
        m.bulge.n.fixed=True
        m.bulge.PA.fixed=True
        m.bulge.ell.fixed=True
        m.disk.h.fixed=True
        m.disk.PA.fixed=True
        m.disk.ell.fixed=True
        models.append(m)
    return models
################################################################################


################################################################################
parser = argparse.ArgumentParser(description='Perform Bulge/Disk decomposition.')

parser.add_argument('galaxyId', type=str, nargs=1,
                    help='CALIFA galaxy ID. Ex.: K0001')
parser.add_argument('runId', type=str, nargs=1,
                    help='runId string. Ex.: eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61')
parser.add_argument('--decomp-id', dest='decompId', default='decomposition',
                    help='Decomposition label.')
parser.add_argument('--db', dest='db', default='../cubes.200/',
                    help='QALIFA database path.')
parser.add_argument('--db-out', dest='dbOutput', default='decomposition.005.h5',
                    help='Output HDF5 database path.')
parser.add_argument('--zone-file-dir', dest='zoneFileDir', default='data/planes',
                    help='Output QBICK-like multiplane FITS image directory.')
parser.add_argument('--mask-file', dest='maskFile', default='data/starlight/Mask.mE',
                    help='Masked wavelengths while performing first pass fit.')
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='Enable verbose output.')
parser.add_argument('--use-fsyn', dest='useFsyn', action='store_true',
                    help='Use synthetic spectra instead of observed.')
parser.add_argument('--box-radius', dest='boxRadius', type=int, default=0,
                    help='Spectral running average box radius.')
parser.add_argument('--box-step', dest='boxStep', type=int, default=1,
                    help='Spectral running average box step.')
parser.add_argument('--vd', dest='vd', type=float, default=None,
                    help='Target v_d in km/s.')
parser.add_argument('--psf-fwhm', dest='psfFWHM', type=float, default=3.6,
                    help='PSF FWHM in arcseconds.')
parser.add_argument('--psf-beta', dest='psfBeta', type=float, default=-1,
                    help='PSF beta parameter for Moffat profile. If not set, use Gaussian.')
parser.add_argument('--psf-size', dest='psfSize', type=int, default=15,
                    help='PSF size, in pixels. Must be an odd number.')
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help='Overwrite data.')
parser.add_argument('--nproc', dest='nproc', type=int, default=None,
                    help='Number of processors to use.')
parser.add_argument('--multipass', dest='multipass', action='store_true',
                    help='Use multiple passes and smoothing.')
parser.add_argument('--smooth-radius', dest='smoothRadius', type=int, default=20,
                    help='Radius of smoothing kernel, if multipass is enabled.')

args = parser.parse_args()
galaxyId = args.galaxyId[0]
# HACK: remove the dots in run name.
runId = args.runId[0]
groupId = runId.replace('.', '_')    

if args.verbose:
    logger.setLevel(-1)
    logger.debug('Verbose output enabled.')

dbfile = path.join(args.db, '%s_synthesis_%s.fits' % (galaxyId, runId))

logger.info('Starting fit for %s...' % galaxyId)
decomp = BulgeDiskDecomposition(dbfile, target_vd=args.vd, use_fobs=not args.useFsyn,
                                PSF_FWHM=args.psfFWHM, PSF_beta=args.psfBeta, PSF_size=args.psfSize,
                                nproc=args.nproc)

if not args.multipass:
    t1 = time.time()
    models = decomp.fitSpectra(step=args.boxStep, box_radius=args.boxRadius)
    logger.info('Done modeling, time: %.2f' % (time.time() - t1))
else:
    t1 = time.time()
    if not path.exists(args.maskFile):
        logger.error('Mask file %s not found.' % args.maskFile)
        exit(1)

    logger.info('Using mask file %s.' % args.maskFile)
    t = atpy.Table(args.maskFile, type='starlight_mask')
    masked_wl = np.zeros(decomp.l_obs.shape, dtype='bool')
    for i in xrange(len(t)):
        l_low, l_upp, line_w, line_name = t[i]
        if line_w > 0.0: continue
        logger.info('Masking region: %s' % line_name)
        masked_wl |= (decomp.l_obs > l_low) & (decomp.l_obs < l_upp)
        
    models = decomp.fitSpectra(step=50*args.boxStep, box_radius=50*args.boxStep, mode='NM', masked_wl=masked_wl)
    first_pass_params = np.array([m.getParams() for m in models], dtype=models[0].dtype)
    logger.info('Done first pass modeling, time: %.2f' % (time.time() - t1))

    t1 = time.time()
    logger.info('Smoothing parameters.')
    models = smooth_models(models, decomp.l_obs)
    
    logger.info('Starting second pass modeling...')
    models = decomp.fitSpectra(step=args.boxStep, box_radius=1,
                               initial_model=models, mode='LM', insist=True)
    logger.info('Done second pass modeling, time: %.2f' % (time.time() - t1))

t1 = time.time()
logger.info('Computing model spectra...')

f_bulge__lyx, f_disk__lyx = decomp.getModelSpectra(models)

# TODO: better array and dtype handling.
fit_params = np.array([m.getParams() for m in models], dtype=models[0].dtype)

Nl_obs = decomp.getNSlices(args.boxStep)
shape__l = (Nl_obs,)
shape__lz = (Nl_obs, decomp.N_zone)
shape__lyx = (Nl_obs, decomp.N_y, decomp.N_x)
dtype = np.dtype([('f_obs', 'float64'), ('f_syn', 'float64'), ('f_err', 'float64'), ('f_flag', 'float64')])
fit_l_obs = decomp.l_obs[::args.boxStep]
flag_bad_fit = fit_params['flag'][:, np.newaxis] > 0.0

# Zonify component fluxes and ratios.
f__lz = np.empty(shape=shape__lz, dtype=dtype)
f__lz['f_obs'] = decomp.f_obs_rest__lz[::args.boxStep]
f__lz['f_syn'] = decomp.f_syn_rest__lz[::args.boxStep]
f__lz['f_err'] = decomp.f_err_rest__lz[::args.boxStep]
f__lz['f_flag'] = decomp.f_flag_rest__lz[::args.boxStep]

f_bulge__lz = np.zeros(shape=shape__lz, dtype=dtype)
r_bulge__lz = np.zeros(shape=shape__lz)
if args.useFsyn:
    f_bulge__lz['f_syn'] = decomp.YXToZone(f_bulge__lyx, extensive=True, surface_density=False)
    good = f__lz['f_syn'] > 0
    r_bulge__lz[good] = f_bulge__lz['f_syn'][good] / f__lz['f_syn'][good]
    f_bulge__lz['f_obs'] = r_bulge__lz * f__lz['f_obs']
else:
    f_bulge__lz['f_obs'] = decomp.YXToZone(f_bulge__lyx, extensive=True, surface_density=False)
    good = f__lz['f_obs'] > 0
    r_bulge__lz[good] = f_bulge__lz['f_obs'][good] / f__lz['f_obs'][good]
    f_bulge__lz['f_syn'] = r_bulge__lz * f__lz['f_syn']
f_bulge__lz['f_err'][good] = np.sqrt(r_bulge__lz[good]) * f__lz['f_err'][good]
f_bulge__lz['f_flag'] = f__lz['f_flag']
f_bulge__lz['f_flag'] += flag_big_error(f_bulge__lz['f_obs'], f_bulge__lz['f_err'])
f_bulge__lz['f_flag'] += flag_small_error(f_bulge__lz['f_obs'], f_bulge__lz['f_err'], f_bulge__lz['f_flag'])
f_bulge__lz['f_flag'] += flag_bad_fit

f_disk__lz = np.zeros(shape=shape__lz, dtype=dtype)
r_disk__lz = np.zeros(shape=shape__lz)
if args.useFsyn:
    f_disk__lz['f_syn'] = decomp.YXToZone(f_disk__lyx, extensive=True, surface_density=False)
    good = f__lz['f_syn'] > 0
    r_disk__lz[good] = f_disk__lz['f_syn'][good] / f__lz['f_syn'][good]
    f_disk__lz['f_obs'] = r_disk__lz * f__lz['f_obs']
else:
    f_disk__lz['f_obs'] = decomp.YXToZone(f_disk__lyx, extensive=True, surface_density=False)
    good = f__lz['f_obs'] > 0
    r_disk__lz[good] = f_disk__lz['f_obs'][good] / f__lz['f_obs'][good]
    f_disk__lz['f_syn'] = r_disk__lz * f__lz['f_syn']
pos_err = r_disk__lz > 0
f_disk__lz['f_err'][pos_err] = np.sqrt(r_disk__lz[pos_err]) * f__lz['f_err'][pos_err]
f_disk__lz['f_flag'] = f__lz['f_flag']
f_disk__lz['f_flag'] += flag_big_error(f_disk__lz['f_obs'], f_disk__lz['f_err'])
f_disk__lz['f_flag'] += flag_small_error(f_disk__lz['f_obs'], f_disk__lz['f_err'], f_disk__lz['f_flag'])
f_disk__lz['f_flag'] += flag_bad_fit

# Integrated spectra
i_f__l = np.empty(shape=shape__l, dtype=dtype)
i_f__l['f_syn'] = f__lz['f_syn'].sum(axis=1)
i_f__l['f_obs'], i_f__l['f_err'], i_f__l['f_flag'] = integrated_spec(f__lz['f_obs'],
                                                                     f__lz['f_err'],
                                                                     f__lz['f_flag'])

i_f_bulge__l = np.empty(shape=shape__l, dtype=dtype)
i_f_bulge__l['f_syn'] = f_bulge__lz['f_syn'].sum(axis=1)
i_f_bulge__l['f_obs'], i_f_bulge__l['f_err'], i_f_bulge__l['f_flag'] = integrated_spec(f_bulge__lz['f_obs'],
                                                                                       f_bulge__lz['f_err'],
                                                                                       f_bulge__lz['f_flag'])

i_f_disk__l = np.empty(shape=shape__l, dtype=dtype)
i_f_disk__l['f_syn'] = f_disk__lz['f_syn'].sum(axis=1)
i_f_disk__l['f_obs'], i_f_disk__l['f_err'], i_f_disk__l['f_flag'] = integrated_spec(f_disk__lz['f_obs'],
                                                                                    f_disk__lz['f_err'],
                                                                                    f_disk__lz['f_flag'])


logger.info('Creating qbick planes...')
l_mask = np.where((fit_l_obs > 5590.0) & (fit_l_obs < 5680.0))[0]
total_planes = get_planes_image(fit_l_obs, f__lz, l_mask, decomp)
save_qbick_planes(total_planes, decomp,
                  path.join(args.zoneFileDir, '%s_%s_%s-total-planes.fits' % (galaxyId, runId, args.decompId)))
bulge_planes = get_planes_image(fit_l_obs, f_bulge__lz, l_mask, decomp)
save_qbick_planes(bulge_planes, decomp,
                  path.join(args.zoneFileDir, '%s_%s_%s-bulge-planes.fits' % (galaxyId, runId, args.decompId)))
disk_planes = get_planes_image(fit_l_obs, f_disk__lz, l_mask, decomp)
save_qbick_planes(disk_planes, decomp,
                  path.join(args.zoneFileDir, '%s_%s_%s-disk-planes.fits' % (galaxyId, runId, args.decompId)))


logger.info('Saving to storage...')
db = openFile(args.dbOutput, 'a')
try:
    grp = db.getNode('/%s/%s/%s' % (args.decompId, groupId, galaxyId))
except:
    grp = db.createGroup('/%s/%s' % (args.decompId, groupId), galaxyId, createparents=True)
    
if args.overwrite and 'fit_parameters' in grp:
    grp.fit_parameters._f_remove()

t = db.createTable(grp, 'fit_parameters', fit_params.dtype, 'Morphology fit parameters', Filters(1, 'blosc'),
              expectedrows=len(fit_params))
t.attrs.FWHM = args.psfFWHM
t.attrs.PSF_FWHM = args.psfFWHM
t.attrs.PSF_beta = args.psfBeta
t.attrs.PSF_size = args.psfSize
t.attrs.box_step = args.boxStep
t.attrs.box_radius = args.boxRadius
t.attrs.orig_file = dbfile
t.attrs.object_name = decomp.galaxyName
t.attrs.flux_unit = decomp.flux_unit
t.attrs.distance_Mpc = decomp.distance_Mpc
t.attrs.x0 = decomp.x0
t.attrs.y0 = decomp.y0
t.attrs.use_fsyn = args.useFsyn
t.attrs.target_vd = decomp.target_vd
t.append(fit_params)
t.flush()

if args.overwrite and 'first_pass_parameters' in grp:
    grp.first_pass_parameters._f_remove()

if args.multipass:
    t = db.createTable(grp, 'first_pass_parameters', fit_params.dtype, 'Morphology first pass fit parameters', Filters(1, 'blosc'),
              expectedrows=len(first_pass_params))
    t.append(first_pass_params)
    t.flush()

save_array(db, grp, 'qMask', decomp.qMask, args.overwrite)
save_array(db, grp, 'qSignal', decomp.qSignal, args.overwrite)
save_array(db, grp, 'qZones', decomp.qZones, args.overwrite)

save_compound_array(db, grp, 'qbick_planes', total_planes, args.overwrite)
save_compound_array(db, grp, 'qbick_planes_bulge', bulge_planes, args.overwrite)
save_compound_array(db, grp, 'qbick_planes_disk', disk_planes, args.overwrite)

save_compound_array(db, grp, 'f__lz', f__lz, args.overwrite)
save_compound_array(db, grp, 'f_bulge__lz', f_bulge__lz, args.overwrite)
save_compound_array(db, grp, 'f_disk__lz', f_disk__lz, args.overwrite)

save_compound_array(db, grp, 'i_f__l', i_f__l, args.overwrite)
save_compound_array(db, grp, 'i_f_bulge__l', i_f_bulge__l, args.overwrite)
save_compound_array(db, grp, 'i_f_disk__l', i_f_disk__l, args.overwrite)

save_array(db, grp, 'f_syn__lyx', decomp.f_syn_rest__lyx[::args.boxStep], args.overwrite)
save_array(db, grp, 'f_obs__lyx', decomp.f_obs_rest__lyx[::args.boxStep], args.overwrite)
save_array(db, grp, 'f_bulge__lyx', f_bulge__lyx, args.overwrite)
save_array(db, grp, 'f_disk__lyx', f_disk__lyx, args.overwrite)

db.close()

logger.info('Storage complete, time: %.2f' % (time.time() - t1))

