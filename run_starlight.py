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



###############################################################################
def write_table(wl, flux, filename):
    ascii.write([wl, flux], filename, Writer=ascii.NoHeader)
###############################################################################


###############################################################################
class GridManager(object):
    
    def __init__(self, starlight_dir, decomp_file, decomp_id, run_id, galaxy_id):
        self.starlightDir = starlight_dir
        self.basesDir = path.join(starlight_dir, 'BasesDir')
        self.specFileDir = path.join(starlight_dir, 'input')
        self.outDir = path.join(starlight_dir, 'output')
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
        grid.basesDir = path.abspath(self.basesDir) + path.sep
        grid.obsDir = path.abspath(self.specFileDir) + path.sep
        grid.maskDir = path.abspath(self.starlightDir) + path.sep
        grid.etcDir = path.abspath(self.starlightDir) + path.sep
        grid.outDir = path.abspath(self.outDir) + path.sep
        grid.fluxUnit = self.flux_unit
        run_template = grid.runs.pop()
        for z in xrange(zone1, zone2):
            if z >= self.N_zone:
                break
            print 'Created inputs for zone %d' % z
    
            new_run = GridRun.sameAs(run_template)
            new_run.inFile = '%s_%04d_%s.syn.in' % (self.galaxyId, z, self.runId)
            new_run.outFile = '%s_%04d_%s.syn.out' % (self.galaxyId, z, self.runId)
            new_run.lumDistanceMpc = self.distance_Mpc
            write_table(self.wl, self.f_syn[:, z], path.join(grid.obsDir, new_run.inFile))
            grid.runs.append(new_run)

            new_run = GridRun.sameAs(run_template)
            new_run.inFile = '%s_%04d_%s.disk.in' % (self.galaxyId, z, self.runId)
            new_run.outFile = '%s_%04d_%s.disk.out' % (self.galaxyId, z, self.runId)
            new_run.lumDistanceMpc = self.distance_Mpc
            write_table(self.wl, self.f_disk[:, z], path.join(grid.obsDir, new_run.inFile))
            grid.runs.append(new_run)

            new_run = GridRun.sameAs(run_template)
            new_run.inFile = '%s_%04d_%s.bulge.in' % (self.galaxyId, z, self.runId)
            new_run.outFile = '%s_%04d_%s.bulge.out' % (self.galaxyId, z, self.runId)
            new_run.lumDistanceMpc = self.distance_Mpc
            write_table(self.wl, self.f_bulge[:, z], path.join(grid.obsDir, new_run.inFile))
            grid.runs.append(new_run)

            new_run = GridRun.sameAs(run_template)
            new_run.inFile = '%s_%04d_%s.res.in' % (self.galaxyId, z, self.runId)
            new_run.outFile = '%s_%04d_%s.res.out' % (self.galaxyId, z, self.runId)
            new_run.lumDistanceMpc = self.distance_Mpc
            residual = self.f_syn[:, z] - self.f_bulge[:, z] - self.f_disk[:, z]
            write_table(self.wl, residual, path.join(grid.obsDir, new_run.inFile))
            grid.runs.append(new_run)
        return grid


    def getGrids(self, chunk_size):
        for z in xrange(0, self.N_zone, chunk_size):
            yield self._getGrid(z, z + chunk_size)

###############################################################################


#sr.starlight_exec_path = '/Users/andre/astro/qalifa/pystarlight/src/pystarlight/mock/mock_starlight.py'
#sr.checker_exec_path = '/Users/andre/astro/starlight/starlight_checker'

starlight_dir = 'data/starlight'
galaxy_id = 'K0127'
decomp_id = 'all_psf00_nocenter'
run_id = 'eBR_v20_q036.d13c512.ps03.k2.mC.CCM.Bgsd61'


gm = GridManager(starlight_dir, 'data/decomposition.006.h5', decomp_id, run_id, galaxy_id)
# HACK: do not take too long.
gm.N_zone = 4


runner = sr.StarlightRunner(n_workers=1)
for grid in gm.getGrids(chunk_size=1):
    runner.addGrid(grid)
    print 'Dispatched grid'

print 'Waiting jobs completion.'
runner.wait()
print 'Yay!'

