'''
Created on Jun 17, 2013

@author: andre
'''
import tables
from astropy.io import ascii
from pystarlight.util.gridfile import GridFile
import numpy as np
from os import path, makedirs as os_makedirs
from pystarlight.util import starlight_runner as sr
import argparse
from multiprocessing import cpu_count



###############################################################################
def write_table(wl, flux, err, flags, filename):
    ascii.write([wl, flux, err, flags], filename, Writer=ascii.NoHeader)
###############################################################################


###############################################################################
def makedirs(the_path):
    if not path.exists(the_path):
        os_makedirs(the_path)
###############################################################################


###############################################################################
class GridManager(object):
    
    def __init__(self, starlight_dir, decomp_file, decomp_id, run_id, galaxy_id, spec_type='f_obs'):
        self.specType = spec_type
        self.starlightDir = starlight_dir
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
        self._gridTemplate, self._runTemplate = self._getTemplates()
        self._createDirs()

        
    def _getTemplates(self):
        template_path = path.join(self.starlightDir, 'grid.template.in')
        grid = GridFile.fromFile(self.starlightDir, template_path)
        grid.setLogDir('log') 
        grid.fluxUnit = self.flux_unit
        run = grid.runs.pop()
        grid.clearRuns()
        return grid, run

    
    def _createDirs(self):
        obs_dir = path.normpath(self._gridTemplate.obsDirAbs)
        out_dir = path.normpath(self._gridTemplate.outDirAbs)
        log_dir = path.normpath(self._gridTemplate.logDirAbs)

        makedirs(obs_dir + '_bulge')
        makedirs(obs_dir + '_disk')
        makedirs(obs_dir + '_total')
        
        makedirs(out_dir + '_bulge')
        makedirs(out_dir + '_disk')
        makedirs(out_dir + '_total')
        
        makedirs(log_dir + '_bulge')
        makedirs(log_dir + '_disk')
        makedirs(log_dir + '_total')
        
        
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
        
        
    def _getGrid(self, component, spectra, zone1, zone2):
        grid = self._gridTemplate.copy()
        if zone1 != zone2:
            grid.name = 'grid_%04d-%04d' % (zone1, zone2)
        else:
            grid.name = 'grid_%04d' % zone1
        grid.setObsDir(grid.obsDir + '_%s' % component)
        grid.setOutDir(grid.outDir + '_%s' % component)
        grid.setLogDir(grid.logDir + '_%s' % component)
        grid.randPhone = -958089828
        # grid.seed()
        
        for z in xrange(zone1, zone2):
            if z >= self.N_zone:
                break
            print 'Creating inputs for %s, zone %d' % (component, z)
            run = self._createRun(component, spectra, z, grid.obsDirAbs)
            if run is not None:
                grid.runs.append(run)
            else:
                print 'Skipping zone %d of %s' % (z, component)
        return grid


    def _createRun(self, component, spectra, z, obs_dir):
        n_good = (spectra[:, z]['f_flag'] == 0.0).sum()
        if n_good <= 400:
            return None
        new_run = self._runTemplate.copy()
        new_run.inFile = '%s_%04d_%s.%s.%s.in' % (self.galaxyId, z, self.runId, self.specType, component)
        new_run.outFile = '%s_%04d_%s.%s.%s.out' % (self.galaxyId, z, self.runId, self.specType, component)
        new_run.lumDistanceMpc = self.distance_Mpc
        write_table(self.l_obs, spectra[:, z][self.specType],
                    spectra[:, z]['f_err'], spectra[:, z]['f_flag'],
                    path.join(obs_dir, new_run.inFile))
        return new_run


    def gridIterator(self, chunk_size):
        for z in xrange(0, self.N_zone, chunk_size):
            yield self._getGrid('bulge', self.f__lz, z, z + chunk_size)
        for z in xrange(0, self.N_zone, chunk_size):
            yield self._getGrid('disk', self.f__lz, z, z + chunk_size)
        for z in xrange(0, self.N_zone, chunk_size):
            yield self._getGrid('total', self.f__lz, z, z + chunk_size)

###############################################################################


#sr.starlight_exec_path = '/Users/andre/astro/qalifa/pystarlight/src/pystarlight/mock/mock_starlight.py'

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
gm = GridManager(args.starlightDir, args.db, args.decompId, run_id, galaxy_id, spec_type=args.spectraType)
print 'Number of zones: %d' % gm.N_zone
print 'Starting starlight runner.'
runner = sr.StarlightRunner(n_workers=nproc, timeout=args.timeout * 60.0, compress=True)
for grid in gm.gridIterator(chunk_size=args.chunkSize):
    print 'Dispatching grid.'
    runner.addGrid(grid)

print 'Waiting jobs completion.'
runner.wait()

failed_grids = runner.getFailedGrids()
if len(failed_grids) > 0:
    print 'Failed to starlight:'
    for grid in failed_grids:
        print '\n'.join(r.outFile for r in grid.failed)

