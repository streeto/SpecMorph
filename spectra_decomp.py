'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso.util import logger
from specmorph import BulgeDiskDecomposition
from specmorph.model import GalaxyModel
from specmorph.qbick import integrated_spec, flag_big_error, flag_small_error, calc_sn

from tables import openFile, Filters
import numpy as np

from os import path
import argparse
import time
from tables.atom import Atom
import pyfits


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
    hdulist = pyfits.HDUList([phdu])
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
def smooth_param_polynomial(param, param_l_obs, param_flags, l_obs, degree=1):
    flag_ok = (param_flags == 0)
    from astropy.modeling import models, fitting
    line = models.Polynomial1D(degree)
    fit = fitting.LinearLSQFitter()
    param_fitted = fit(line, param_l_obs[flag_ok], param[flag_ok])
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.plot(param_l_obs[flag_ok], param[flag_ok], 'ok')
    plt.plot(l_obs, param_fitted(l_obs), '-r')
    plt.show()
    return param_fitted(l_obs)
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
                    help='Output HDF5 database path.')
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='Enable verbose output.')
parser.add_argument('--box-radius', dest='boxRadius', type=int, default=0,
                    help='Spectral running average box radius.')
parser.add_argument('--box-step', dest='boxStep', type=int, default=1,
                    help='Spectral running average box step.')
parser.add_argument('--psf-fwhm', dest='psfFWHM', type=float, default=3.6,
                    help='PSF FWHM in arcseconds.')
parser.add_argument('--psf-beta', dest='psfBeta', type=float, default=-1,
                    help='PSF beta parameter for Moffat profile. If not set, use Gaussian.')
parser.add_argument('--psf-size', dest='psfSize', type=int, default=15,
                    help='PSF size, in pixels. Must be an odd number.')
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help='Overwrite data.')
parser.add_argument('--nproc', dest='nproc', type=int, default=-1,
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
decomp = BulgeDiskDecomposition(dbfile, target_vd=300.0,
                                PSF_FWHM=args.psfFWHM, PSF_beta=args.psfBeta, PSF_size=args.psfSize,
                                nproc=args.nproc)

if not args.multipass:
    t1 = time.time()
    models, fit_l_ix = decomp.fitSpectra(step=args.boxStep, box_radius=args.boxRadius)
    logger.info('Done modeling, time: %.2f' % (time.time() - t1))
else:
    t1 = time.time()
    models, fit_l_ix = decomp.fitSpectra(step=50, box_radius=50, mode='NM')
    logger.info('Done first pass modeling, time: %.2f' % (time.time() - t1))
    t1 = time.time()
    logger.info('Smoothing parameters.')
    
    first_pass_params = np.array([m.getParams() for m in models], dtype=models[0].dtype)
    initial_params = np.empty(len(decomp.l_obs), dtype=first_pass_params.dtype)

    flags = first_pass_params['flag']
    first_pass_l_obs = decomp.l_obs[fit_l_ix]
    
    for p in first_pass_params.dtype.names:
        if p in ['flag', 'chi2', 'n_pix']: continue
        initial_params[p] = smooth_param_polynomial(first_pass_params[p], first_pass_l_obs, flags,
                                                    decomp.l_obs, degree=1)
    
    models = []
    for p in initial_params:
        m = GalaxyModel.fromParamVector(p)
        m.x0.setValue(p['x0'], fixed=True)
        m.y0.setValue(p['y0'], fixed=True)
        m.bulge.r_e.setValue(p['r_e'], fixed=True)
        m.bulge.n.setValue(p['n'], fixed=True)
        m.bulge.PA.setValue(p['PA_b'], fixed=True)
        m.bulge.ell.setValue(p['ell_b'], fixed=True)
        m.disk.h.setValue(p['h'], fixed=True)
        m.disk.PA.setValue(p['PA_d'], fixed=True)
        m.disk.ell.setValue(p['ell_d'], fixed=True)
        models.append(m)
    
    logger.info('Starting second pass modeling...')
    models, fit_l_ix = decomp.fitSpectra(step=args.boxStep, box_radius=args.boxRadius,
                                         initial_model=models, mode='LM', insist=True)
    logger.info('Done second pass modeling, time: %.2f' % (time.time() - t1))
    

t1 = time.time()
logger.info('Computing model spectra...')

f_syn_bulge__lyx, f_syn_disk__lyx = decomp.getModelSpectra(models)

# TODO: better array and dtype handling.
fit_params = np.array([m.getParams() for m in models], dtype=models[0].dtype)

shape__l = (len(fit_l_ix),)
shape__lz = (len(fit_l_ix), decomp.N_zone)
shape__lyx = (len(fit_l_ix), decomp.N_y, decomp.N_x)
dtype = np.dtype([('f_obs', 'float64'), ('f_syn', 'float64'), ('f_err', 'float64'), ('f_flag', 'float64')])
fit_l_obs = decomp.l_obs[fit_l_ix]
# FIXME: only flagging fits that stepped out of bounds.
flag_bad_fit = fit_params['flag'][:, np.newaxis] > 0.0

f__lz = np.empty(shape=shape__lz, dtype=dtype)
f__lz['f_obs'] = decomp.f_obs_rest__lz[fit_l_ix]
f__lz['f_syn'] = decomp.f_syn_rest__lz[fit_l_ix]
f__lz['f_err'] = decomp.f_err_rest__lz[fit_l_ix]
f__lz['f_flag'] = decomp.f_flag_rest__lz[fit_l_ix]

# Zonify component fluxes and ratios.
f_syn_bulge__lz = decomp.YXToZone(f_syn_bulge__lyx, extensive=True, surface_density=False)
f_syn_disk__lz = decomp.YXToZone(f_syn_disk__lyx, extensive=True, surface_density=False)
r_bulge__lz = f_syn_bulge__lz / f__lz['f_syn']
r_disk__lz = f_syn_disk__lz / f__lz['f_syn']

f_bulge__lz = np.empty(shape=shape__lz, dtype=dtype)
f_bulge__lz['f_obs'] = r_bulge__lz * f__lz['f_obs']
f_bulge__lz['f_syn'] = f_syn_bulge__lz
f_bulge__lz['f_err'] = np.sqrt(r_bulge__lz) * f__lz['f_err']
f_bulge__lz['f_flag'] = f__lz['f_flag']
f_bulge__lz['f_flag'] += flag_big_error(f_bulge__lz['f_obs'], f_bulge__lz['f_err'])
f_bulge__lz['f_flag'] += flag_small_error(f_bulge__lz['f_obs'], f_bulge__lz['f_err'], f_bulge__lz['f_flag'])
f_bulge__lz['f_flag'] += flag_bad_fit

f_disk__lz = np.empty(shape=shape__lz, dtype=dtype)
f_disk__lz['f_obs'] = r_disk__lz * f__lz['f_obs']
f_disk__lz['f_syn'] = f_syn_disk__lz
f_disk__lz['f_err'] = np.sqrt(r_disk__lz) * f__lz['f_err']
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
save_qbick_planes(total_planes, decomp, path.join(args.zoneFileDir, '%s_%s_%s-total-planes.fits' % (galaxyId, runId, args.decompId)))
bulge_planes = get_planes_image(fit_l_obs, f_bulge__lz, l_mask, decomp)
save_qbick_planes(bulge_planes, decomp, path.join(args.zoneFileDir, '%s_%s_%s-bulge-planes.fits' % (galaxyId, runId, args.decompId)))
disk_planes = get_planes_image(fit_l_obs, f_disk__lz, l_mask, decomp)
save_qbick_planes(disk_planes, decomp, path.join(args.zoneFileDir, '%s_%s_%s-disk-planes.fits' % (galaxyId, runId, args.decompId)))


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
t.append(fit_params)
t.flush()

if args.multipass:
    t = db.createTable(grp, 'first_pass_parameters', fit_params.dtype, 'Morphology fisrt pass fit parameters', Filters(1, 'blosc'),
              expectedrows=len(first_pass_params))
    t.append(first_pass_params)
    t.flush()
    save_array(db, grp, 'first_pass_l_obs', first_pass_l_obs, args.overwrite)

save_array(db, grp, 'qMask', decomp.qMask, args.overwrite)
save_array(db, grp, 'qSignal', decomp.qSignal, args.overwrite)
save_array(db, grp, 'qZones', decomp.qZones, args.overwrite)

save_compound_array(db, grp, 'qbick_planes', total_planes, args.overwrite)
save_compound_array(db, grp, 'qbick_planes_bulge', bulge_planes, args.overwrite)
save_compound_array(db, grp, 'qbick_planes_disk', disk_planes, args.overwrite)

save_array(db, grp, 'l_obs', fit_l_obs, args.overwrite)

save_compound_array(db, grp, 'f__lz', f__lz, args.overwrite)
save_compound_array(db, grp, 'f_bulge__lz', f_bulge__lz, args.overwrite)
save_compound_array(db, grp, 'f_disk__lz', f_disk__lz, args.overwrite)

save_compound_array(db, grp, 'i_f__l', i_f__l, args.overwrite)
save_compound_array(db, grp, 'i_f_bulge__l', i_f_bulge__l, args.overwrite)
save_compound_array(db, grp, 'i_f_disk__l', i_f_disk__l, args.overwrite)

save_array(db, grp, 'f_syn__lyx', decomp.f_syn_rest__lyx, args.overwrite)
save_array(db, grp, 'f_syn_bulge__lyx', f_syn_bulge__lyx, args.overwrite)
save_array(db, grp, 'f_syn_disk__lyx', f_syn_disk__lyx, args.overwrite)

db.close()

logger.info('Storage complete, time: %.2f' % (time.time() - t1))

