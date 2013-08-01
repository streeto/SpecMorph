'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso.util import logger
from specmorph import BulgeDiskDecomposition
from specmorph.qbick import integrated_spec, flag_big_error, flag_small_error

from tables import openFile, Filters
import numpy as np

from os import path
import argparse
import time
from tables.atom import Atom


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
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='Enable verbose output.')
parser.add_argument('--box-radius', dest='boxRadius', type=int, default=0,
                    help='Spectral running average box radius.')
parser.add_argument('--box-step', dest='boxStep', type=int, default=1,
                    help='Spectral running average box step.')
parser.add_argument('--rad-clip', dest='radClip', type=float, default=0.0,
                    help='Radial clip in arc seconds (float).')
parser.add_argument('--psf-fwhm', dest='fwhm', type=float, default=3.6,
                    help='PSF FWHM in arcseconds.')
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help='Overwrite data.')
parser.add_argument('--nproc', dest='nproc', type=int, default=-1,
                    help='Number of processors to use.')

args = parser.parse_args()
galaxyId = args.galaxyId[0]
# HACK: remove the dots in run name.
runId = args.runId[0]
groupId = runId.replace('.', '_')    

if args.verbose:
    logger.setLevel(-1)
    logger.debug('Verbose output enabled.')

dbfile = path.join(args.db, '%s_synthesis_%s.fits' % (galaxyId, runId))

t1 = time.time()
logger.info('Starting fit for %s...' % galaxyId)
decomp = BulgeDiskDecomposition(dbfile, target_vd=0.0, FWHM=args.fwhm, nproc=args.nproc)
fit_params, fit_l_ix = decomp.fitSpectra(step=args.boxStep, box_radius=args.boxRadius,
                                         rad_clip_in=args.radClip, rad_clip_out=None, mode='mean')
logger.info('Done modeling, time: %.2f' % (time.time() - t1))

t1 = time.time()
logger.info('Computing model spectra...')

f_syn_bulge__lyx, f_syn_disk__lyx = decomp.getModelSpectra(fit_params)

shape__l = (len(fit_l_ix),)
shape__lz = (len(fit_l_ix), decomp.N_zone)
shape__lyx = (len(fit_l_ix), decomp.N_y, decomp.N_x)
dtype = np.dtype([('f_obs', 'float64'), ('f_syn', 'float64'), ('f_err', 'float64'), ('f_flag', 'float64')])
fit_l_obs = decomp.l_obs[fit_l_ix]

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

f_disk__lz = np.empty(shape=shape__lz, dtype=dtype)
f_disk__lz['f_obs'] = r_disk__lz * f__lz['f_obs']
f_disk__lz['f_syn'] = f_syn_disk__lz
f_disk__lz['f_err'] = np.sqrt(r_disk__lz) * f__lz['f_err']
f_disk__lz['f_flag'] = f__lz['f_flag']
f_disk__lz['f_flag'] += flag_big_error(f_disk__lz['f_obs'], f_disk__lz['f_err'])
f_disk__lz['f_flag'] += flag_small_error(f_disk__lz['f_obs'], f_disk__lz['f_err'], f_disk__lz['f_flag'])


# Integrated spectra
i_f__l = np.empty(shape=shape__l, dtype=dtype)
i_f__l['f_syn'] = f__lz['f_syn'].sum(axis=1)
i_f__l['f_obs'], i_f__l['f_err'], i_f__l['f_flag'] = integrated_spec(f__lz['f_obs'],
                                                                     f__lz['f_err'],
                                                                     f__lz['f_flag'])

# FIXME: bulge central pixels are not OK if we don't use
# the correct PSF when fitting. Discard the central zones.
zonedist = np.sqrt((decomp.zonePos['x'] - decomp.x0)**2 + (decomp.zonePos['y'] - decomp.y0)**2)
bulge_ok = zonedist >= args.radClip
i_f_bulge__l = np.empty(shape=shape__l, dtype=dtype)
i_f_bulge__l['f_syn'] = f_bulge__lz['f_syn'][:,bulge_ok].sum(axis=1)
i_f_bulge__l['f_obs'], i_f_bulge__l['f_err'], i_f_bulge__l['f_flag'] = integrated_spec(f_bulge__lz['f_obs'][:,bulge_ok],
                                                                                       f_bulge__lz['f_err'][:,bulge_ok],
                                                                                       f_bulge__lz['f_flag'][:,bulge_ok])

i_f_disk__l = np.empty(shape=shape__l, dtype=dtype)
i_f_disk__l['f_syn'] = f_disk__lz['f_syn'].sum(axis=1)
i_f_disk__l['f_obs'], i_f_disk__l['f_err'], i_f_disk__l['f_flag'] = integrated_spec(f_disk__lz['f_obs'],
                                                                                    f_disk__lz['f_err'],
                                                                                    f_disk__lz['f_flag'])


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
t.attrs.FWHM = args.fwhm
t.attrs.box_step = args.boxStep
t.attrs.box_radius = args.boxRadius
t.attrs.rad_clip = args.radClip
t.attrs.orig_file = dbfile
t.attrs.object_name = decomp.galaxyName
t.attrs.flux_unit = decomp.flux_unit
t.attrs.distance_Mpc = decomp.distance_Mpc
t.attrs.x0 = decomp.x0
t.attrs.y0 = decomp.y0
t.append(fit_params)
t.flush()

def save_compound_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        print 'Removing existing group %s' % name
        parent._f_getChild(name)._f_remove(recursive=True)
    grp = db.createGroup(parent, name)
    # HACK: pytables does not support compound dtypes.
    for field in data.dtype.names:
        save_array(db, grp, field, data[field], overwrite)

    
def save_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        print 'Removing existing array %s' % name
        parent._f_getChild(name)._f_remove()
    ca = db.createCArray(parent, name, Atom.from_dtype(data.dtype), data.shape, filters=Filters(1, 'blosc'))
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled()
    ca[...] = data
    
save_array(db, grp, 'qMask', decomp.qMask, args.overwrite)
save_array(db, grp, 'qSignal', decomp.qSignal, args.overwrite)
save_array(db, grp, 'qZones', decomp.qZones, args.overwrite)

save_array(db, grp, 'l_obs', fit_l_obs, args.overwrite)

save_compound_array(db, grp, 'f__lz', f__lz, args.overwrite)
save_compound_array(db, grp, 'f_bulge__lz', f_bulge__lz, args.overwrite)
save_compound_array(db, grp, 'f_disk__lz', f_disk__lz, args.overwrite)

save_compound_array(db, grp, 'i_f__l', i_f__l, args.overwrite)
save_compound_array(db, grp, 'i_f_bulge__l', i_f_bulge__l, args.overwrite)
save_compound_array(db, grp, 'i_f_disk__l', i_f_disk__l, args.overwrite)

save_array(db, grp, 'f_syn_bulge__lyx', f_syn_bulge__lyx, args.overwrite)
save_array(db, grp, 'f_syn_disk__lyx', f_syn_disk__lyx, args.overwrite)

db.close()

logger.info('Storage complete, time: %.2f' % (time.time() - t1))

