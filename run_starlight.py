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
def write_table(wl, flux, filename):
    ascii.write([wl, flux], filename, Writer=ascii.NoHeader)
###############################################################################


###############################################################################
class GridManager(object):
    
    def __init__(self, starlight_dir, decomp_file, decomp_id, run_id, galaxy_id):
        self.starlightDir = starlight_dir
        self.basesDir = 'BasesDir'
        self.obsDir = 'input'
        self.outDir = 'output'
        self.maskDir = '.'
        self.etcDir = '.'
        self.galaxyId = galaxy_id
        self.decompId = decomp_id
        self.runId = run_id
        groupId = run_id.replace('.', '_')   
        f = tables.openFile(decomp_file)
        grp = f.getNode('/%s/%s/%s' % (decomp_id,groupId, galaxy_id))
        self.wl = grp.wavelength[:]
        
        t = grp.fit_parameters
        self.flux_unit = t.attrs.flux_unit
        self.distance_Mpc = t.attrs.distance_Mpc
        
        self.N_zone = grp.zone_synth_spectra.shape[1] + 1
        self.Nl_obs = self.wl.shape[0]
        shape = (self.Nl_obs, self.N_zone)
        self.f_syn = np.empty(shape, dtype='float64')
        self.f_bulge = np.empty(shape, dtype='float64')
        self.f_disk = np.empty(shape, dtype='float64')
        
        # Normalize by flux_unit
        self.f_syn[:, 1:] = grp.zone_synth_spectra[...] / self.flux_unit
        self.f_bulge[:, 1:] = grp.zone_bulge_spectra[...] / self.flux_unit
        self.f_disk[:, 1:] = grp.zone_disk_spectra[...] / self.flux_unit
        
        # Integrated spectra
        self.f_syn[:,0] = self.f_syn[:, 1:].sum(axis=1)
        self.f_bulge[:,0] = self.f_bulge[:, 1:].sum(axis=1)
        self.f_disk[:,0] = self.f_disk[:, 1:].sum(axis=1)
        
        f.close()
        
        
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
        
        obsDir = path.join(self.starlightDir, self.obsDir) 
        for z in xrange(zone1, zone2):
            if z >= self.N_zone:
                break
            print 'Created inputs for zone %d' % z
    
            new_run = GridRun.sameAs(run_template)
            new_run.inFile = '%s_%04d_%s.syn.in' % (self.galaxyId, z, self.runId)
            new_run.outFile = '%s_%04d_%s.syn.out' % (self.galaxyId, z, self.runId)
            new_run.lumDistanceMpc = self.distance_Mpc
            write_table(self.wl, self.f_syn[:, z], path.join(obsDir, new_run.inFile))
            grid.runs.append(new_run)

            new_run = GridRun.sameAs(run_template)
            new_run.inFile = '%s_%04d_%s.disk.in' % (self.galaxyId, z, self.runId)
            new_run.outFile = '%s_%04d_%s.disk.out' % (self.galaxyId, z, self.runId)
            new_run.lumDistanceMpc = self.distance_Mpc
            write_table(self.wl, self.f_disk[:, z], path.join(obsDir, new_run.inFile))
            grid.runs.append(new_run)

            new_run = GridRun.sameAs(run_template)
            new_run.inFile = '%s_%04d_%s.bulge.in' % (self.galaxyId, z, self.runId)
            new_run.outFile = '%s_%04d_%s.bulge.out' % (self.galaxyId, z, self.runId)
            new_run.lumDistanceMpc = self.distance_Mpc
            write_table(self.wl, self.f_bulge[:, z], path.join(obsDir, new_run.inFile))
            grid.runs.append(new_run)
        return grid


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

args = parser.parse_args()
galaxy_id = args.galaxyId[0]
run_id = args.runId[0]
nproc = args.nproc if args.nproc > 1 else 1

gm = GridManager(args.starlightDir, args.db, args.decompId, run_id, galaxy_id)
runner = sr.StarlightRunner(n_workers=nproc)
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

