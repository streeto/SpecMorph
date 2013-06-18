'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso.util import logger
from pycasso.morphology import BulgeDiskDecomposition

from tables import openFile, Filters
from tables import Float64Atom  # @UnresolvedImport

import sys
from os import path
import argparse

parser = argparse.ArgumentParser(description='Perform Bulge/Disk decomposition.')

parser.add_argument('galaxyId', type=str, nargs=1,
                    help='CALIFA galaxy ID. Ex.: K0001')
parser.add_argument('runId', type=str, nargs=1,
                    help='runId string. Ex.: eBR_v20.d13c500.ps03.k1.m0.CCM.Bgsd01.v01')
parser.add_argument('--decomp-id', dest='decompId', default= 'decomposition',
                    help='Decomposition label.')
parser.add_argument('--db', dest='db', default= '../../../cubes.200/',
                    help='QALIFA database path.')
parser.add_argument('--db-out', dest='dbOutput', default='decomposition.005.h5',
                    help='Output HDF5 database path.')
parser.add_argument('--verbose', dest='verbose', action='store_true',
                    help='Enable verbose output.')
parser.add_argument('--box-radius', dest='boxRadius', type=int, default=0,
                    help='Spectral running average box radius.')
parser.add_argument('--box-step', dest='boxStep', type=int, default=1,
                    help='Spectral running average box step.')
parser.add_argument('--rad-clip', dest='radClip', type=float, default=2.5,
                    help='Radial clip in arc seconds (float).')
parser.add_argument('--psf-fwhm', dest='fwhm', type=float, default=0.0,
                    help='PSF FWHM in arcseconds.')
parser.add_argument('--overwrite', dest='overwrite', action='store_true',
                    help='Overwrite data.')

args = parser.parse_args()
galaxyId = args.galaxyId[0]
# HACK: remove the dots in run name.
runId = args.runId[0]
groupId = runId.replace('.', '_')    

if args.verbose:
    logger.setLevel(-1)
    logger.debug('Verbose output enabled.')

dbfile = path.join(args.db, '%s_synthesis_%s.fits' % (galaxyId, runId))

# TODO: find target_vd for all galaxies
decomp = BulgeDiskDecomposition(dbfile, target_vd=0.0)
fit_params, fit_l_ix = decomp.fitSpectra(step=args.boxStep, box_radius=args.boxRadius,
                                         FWHM=args.fwhm, rad_clip_in=args.radClip, rad_clip_out=None,
                                         fit_psf=False, mode='mean')
f_bulge, f_disk = decomp.getModelSpectra(fit_params)

# FIXME: find the indices where decomp.l_obs == fit_l_obs
fit_l_obs = decomp.l_obs[fit_l_ix]
f_syn = decomp.f_syn_fixed__lyx[fit_l_ix]

db = openFile(args.dbOutput, 'a')
try:
    grp = db.getNode('/%s/%s/%s' % (args.decompId, groupId, galaxyId))
except:
    grp = db.createGroup('/%s/%s' % (args.decompId, groupId), galaxyId, createparents=True)
    
if args.overwrite and 'fit_parameters' in grp:
    grp.fit_parameters._f_remove()

t = db.createTable(grp, 'fit_parameters', fit_params.dtype, 'Morphology fit parameters', Filters(1, 'blosc'),
              expectedrows=len(fit_params))
t.attrs.box_step = args.boxStep
t.attrs.box_radius = args.boxRadius
t.attrs.rad_clip = args.radClip
t.attrs.orig_file = dbfile
t.attrs.objec_name = decomp.galaxyName
t.attrs.flux_unit = decomp.flux_unit
t.attrs.distance_Mpc = decomp.distance_Mpc
t.append(fit_params)
t.flush()

if args.overwrite and 'wavelength' in grp:
    grp.wavelength._f_remove()
ca = db.createCArray(grp, 'wavelength', Float64Atom(), fit_l_obs.shape, filters=Filters(1, 'blosc'))
ca[...] = fit_l_obs

if args.overwrite and 'synth_spectra' in grp:
    grp.synth_spectra._f_remove()
ca = db.createCArray(grp, 'synth_spectra', Float64Atom(), f_syn.shape, filters=Filters(1, 'blosc'))
ca[...] = f_syn

if args.overwrite and 'bulge_spectra' in grp:
    grp.bulge_spectra._f_remove()
ca = db.createCArray(grp, 'bulge_spectra', Float64Atom(), f_bulge.shape, filters=Filters(1, 'blosc'))
ca[...] = f_bulge

if args.overwrite and 'disk_spectra' in grp:
    grp.disk_spectra._f_remove()
ca = db.createCArray(grp, 'disk_spectra', Float64Atom(), f_disk.shape, filters=Filters(1, 'blosc'))
ca[...] = f_disk

db.close()
