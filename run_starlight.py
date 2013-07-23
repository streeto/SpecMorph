'''
Created on Jun 17, 2013

@author: andre
'''
import tables
from astropy.io import ascii
from pystarlight.util.gridfile import GridFile, GridRun
import numpy as np
from os import path
from pystarlight.util import starlight_runner as sr
import argparse
from multiprocessing import cpu_count



###############################################################################
def write_table(wl, flux, err, flags, filename):
    ascii.write([wl, flux, err, flags], filename, Writer=ascii.NoHeader)
###############################################################################


###############################################################################
class GridManager(object):
    
    def __init__(self, starlight_dir, decomp_file, decomp_id, run_id, galaxy_id, spec_type='f_obs'):
        self.specType = spec_type
        self.starlightDir = starlight_dir
        self.basesDir = 'BasesDir'
        self.obsDir = 'input'
        self.obsDir_fullpath = path.join(self.starlightDir, self.obsDir)
        self.outDir = 'output'
        self.maskDir = '.'
        self.etcDir = '.'
        self.galaxyId = galaxy_id
        self.decompId = decomp_id
        self.runId = run_id
        groupId = run_id.replace('.', '_')   
        f = tables.openFile(decomp_file)
        grp = f.getNode('/%s/%s/%s' % (decomp_id,groupId, galaxy_id))
        self.l_obs = grp.l_obs[:]
        
        t = grp.fit_parameters
        self.flux_unit = t.attrs.flux_unit
        self.distance_Mpc = t.attrs.distance_Mpc
        
        self.N_zone = grp.f__lz.f_obs.shape[1] + 1
        self.Nl_obs = self.l_obs.shape[0]
        
        self.f__lz = self._getZoneSpectra(grp.f__lz, grp.i_f__l)
        self.f_bulge__lz = self._getZoneSpectra(grp.f_bulge__lz, grp.i_f_bulge__l)
        self.f_disk__lz = self._getZoneSpectra(grp.f_disk__lz, grp.i_f_disk__l)

        f.close()

    
    def _getZoneSpectra(self, grp, igrp):
        shape = (self.Nl_obs, self.N_zone)
        dtype = np.dtype([('f_obs', 'float64'), ('f_syn', 'float64'), ('f_err', 'float64'), ('f_flag', 'float64')])
        zspec = np.empty(shape, dtype=dtype)
        zspec[:, 1:]['f_obs'] = grp.f_obs[...] / self.flux_unit
        zspec[:, 1:]['f_syn'] = grp.f_syn[...] / self.flux_unit
        zspec[:, 1:]['f_err'] = grp.f_err[...] / self.flux_unit
        zspec[:, 1:]['f_flag'] = grp.f_flag[...]
        zspec[:,0]['f_obs'] = igrp.f_obs[...] / self.flux_unit
        zspec[:,0]['f_syn'] = igrp.f_syn[...] / self.flux_unit
        zspec[:,0]['f_err'] = igrp.f_err[...] / self.flux_unit
        zspec[:,0]['f_flag'] = igrp.f_flag[...]
        return zspec
        
        
    def _getGrid(self, zone1, zone2):
        grid = GridFile()
        grid_template_file = path.join(self.starlightDir, 'grid.template.in')
        grid.loadFrom(grid_template_file)
        grid.starlightDir = self.starlightDir
        grid.randPhone = -958089828
        # grid.seed()
        grid.basesDir = self.basesDir + path.sep
        grid.obsDir = self.obsDir + path.sep
        grid.maskDir = self.maskDir + path.sep
        grid.etcDir = self.etcDir + path.sep
        grid.outDir = self.outDir + path.sep
        grid.fluxUnit = self.flux_unit
        run_template = grid.runs.pop()
        
        for z in xrange(zone1, zone2):
            if z >= self.N_zone:
                break
            print 'Creating inputs for zone %d' % z
            grid.runs.append(self._createRun(z, 'total', self.f__lz, run_template))
            grid.runs.append(self._createRun(z, 'bulge', self.f_bulge__lz, run_template))
            grid.runs.append(self._createRun(z, 'disk', self.f_bulge__lz, run_template))
        return grid


    def _createRun(self, z, suffix, spectra, run_template):
        new_run = GridRun.sameAs(run_template)
        new_run.inFile = '%s_%04d_%s.%s.%s.in' % (self.galaxyId, z, self.runId, self.specType, suffix)
        new_run.outFile = '%s_%04d_%s.%s.%s.out' % (self.galaxyId, z, self.runId, self.specType, suffix)
        new_run.lumDistanceMpc = self.distance_Mpc
        write_table(self.l_obs, spectra[:, z][self.specType],
                    spectra[:, z]['f_err'], spectra[:, z]['f_flag'],
                    path.join(self.obsDir_fullpath, new_run.inFile))
        return new_run


    def getGrids(self, chunk_size):
        for z in xrange(0, self.N_zone, chunk_size):
            yield self._getGrid(z, z + chunk_size)

###############################################################################


#sr.starlight_exec_path = '/Users/andre/astro/qalifa/pystarlight/src/pystarlight/mock/mock_starlight.py'
#sr.checker_exec_path = '/Users/andre/astro/starlight/starlight_checker'

parser = argparse.ArgumentParser(description='Run starlight for a B/D decomposition.')

parser.add_argument('galaxyId', type=str, nargs=1,
                    help='CALIFA galaxy ID. Ex.: K0001')
parser.add_argument('runId', type=str, nargs=1,
                    help='runId string. Ex.: eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61')
parser.add_argument('--decomp-id', dest='decompId', default= 'all_psf00_nocenter',
                    help='Decomposition label.')
parser.add_argument('--db', dest='db', default='data/decomposition.006.h5',
                    help='HDF5 database path.')
parser.add_argument('--starlight-dir', dest='starlightDir', default='data/starlight',
                    help='HDF5 database path.')
parser.add_argument('--nproc', dest='nproc', type=int, default=cpu_count()-1,
                    help='Number of worker processes.')
parser.add_argument('--chunk-size', dest='chunkSize', type=int, default=1,
                    help='Grid chunk size, defaults to the same as --nproc.')
parser.add_argument('--timeout', dest='timeout', type=int, default=20,
                    help='Timeout of starlight processes, in minutes. Defaults to 20.')
parser.add_argument('--spectra-type', dest='spectraType', default='f_obs',
                    help='Spectra type, defaults to f_obs (observed spectra).')
args = parser.parse_args()
galaxy_id = args.galaxyId[0]
run_id = args.runId[0]
nproc = args.nproc if args.nproc > 1 else 1

print 'Loading grid manager.'
gm = GridManager(args.starlightDir, args.db, args.decompId, run_id, galaxy_id)
print 'Number of zones: %d' % gm.N_zone
print 'Starting starlight runner.'
runner = sr.StarlightRunner(n_workers=nproc, timeout=args.timeout * 60.0)
for grid in gm.getGrids(chunk_size=args.chunkSize):
    print 'Dispatching grid.'
    runner.addGrid(grid)

print 'Waiting jobs completion.'
runner.wait()

failed = runner.getFailedGrids()
if len(failed) > 0:
    print 'Failed to starlight:'
    for grid in failed:
        print '\n'.join(r.outFile for r in grid.runs)

