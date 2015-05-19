'''
Created on Jun 17, 2013

@author: andre
'''

from specmorph.io import DecompContainer
from pystarlight.util.gridfile import GridFile
from pystarlight.util import starlight_runner as sr

from astropy.io import ascii
import numpy as np
from os import path, makedirs as os_makedirs
import argparse
from multiprocessing import cpu_count



###############################################################################
def write_table(wl, flux, err, flags, filename):
    flags = np.where(flags, 1.0, 0.0)
    ascii.write([wl, flux, err, flags], filename, Writer=ascii.NoHeader)
###############################################################################


###############################################################################
def makedirs(the_path):
    if not path.exists(the_path):
        os_makedirs(the_path)
###############################################################################


###############################################################################
class GridManager(object):
    
    def __init__(self, starlight_dir, db, sample, galaxy_id):
        self.starlightDir = starlight_dir
        self.galaxyId = galaxy_id
        self.sample = sample
        self._dc = DecompContainer()
        self._dc.loadHDF5(db, sample, galaxy_id)
        self._gridTemplate, self._runTemplate = self._getTemplates()
        self._createDirs()

        
    def _getTemplates(self):
        template_path = path.join(self.starlightDir, 'grid.template.in')
        grid = GridFile.fromFile(self.starlightDir, template_path)
        grid.setLogDir('log') 
        grid.fluxUnit = self._dc.attrs['flux_unit']
        run = grid.runs.pop()
        run.lumDistanceMpc = self._dc.attrs['distance_Mpc']
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
        
        
    def _getZoneSpectra(self, component):
        comp = getattr(self._dc, component)
        l_obs = comp.wl
        f_obs, f_err, f_flag = comp.asZones(self._dc.zones)
        f_obs = np.hstack((comp.i_f_obs[:, np.newaxis], f_obs))
        f_err = np.hstack((comp.i_f_err[:, np.newaxis], f_err))
        f_flag = np.hstack((comp.i_f_flag[:, np.newaxis], f_flag))
        return l_obs, f_obs, f_err, f_flag
        
        
    def _getGrid(self, component, l_obs, f_obs, f_err, f_flag, zone1, zone2):
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
            print 'Creating inputs for %s, zone %d' % (component, z)
            run = self._createRun(component, l_obs, f_obs, f_err, f_flag, z, grid.obsDirAbs)
            if run is not None:
                grid.runs.append(run)
            else:
                print 'Skipping zone %d of %s' % (z, component)
        return grid


    def _createRun(self, component, l_obs, f_obs, f_err, f_flag, z, obs_dir):
        n_good = (f_flag == 0.0).sum()
        if n_good <= 400:
            return None
        new_run = self._runTemplate.copy()
        new_run.inFile = '%s_%04d_%s.%s.in' % (self.galaxyId, z, self.sample, component)
        new_run.outFile = '%s_%04d_%s.%s.out' % (self.galaxyId, z, self.sample, component)
        write_table(l_obs, f_obs[:, z], f_err[:, z], f_flag[:, z],
                    path.join(obs_dir, new_run.inFile))
        return new_run


    def gridIterator(self, chunk_size):
        for component in ['bulge', 'disk', 'total']:
            l_obs, f_obs, f_err, f_flag = self._getZoneSpectra(component)
            N_zone = f_obs.shape[1]
            print 'Component %s: %d zones in chunks of %d' % (component, N_zone, chunk_size)
            for z in xrange(0, N_zone, chunk_size):
                yield self._getGrid(component, l_obs, f_obs, f_err, f_flag, z, z + chunk_size)

###############################################################################


sr.starlight_exec_path = '/Users/andre/astro/qalifa/pystarlight/src/pystarlight/mock/mock_starlight.py'

parser = argparse.ArgumentParser(description='Run starlight for a B/D decomposition.')

parser.add_argument('galaxyId', type=str, nargs=1,
                    help='CALIFA galaxy ID. Ex.: K0001')
parser.add_argument('--sample', dest='sample', default= 'sample004',
                    help='Sample table.')
parser.add_argument('--db', dest='db', default='data/decomposition.005.h5',
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
nproc = args.nproc if args.nproc > 1 else 1

print 'Loading grid manager.'
gm = GridManager(args.starlightDir, args.db, args.sample, galaxy_id)
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

