'''
Created on 12/08/2014

@author: andre
'''

from specmorph.util import logger
from specmorph.califa import califa_id_str
from specmorph.califa.qbick import flag_big_error, flag_small_error, calc_sn, integrated_spec
from tables import open_file, Filters
from tables.atom import Atom
import numpy as np

__all__ = ['IFSContainer', 'DecompContainer']

################################################################################
class IFSContainer(object):
    
    def __init__(self, wl=None, f_obs=None, f_err=None, f_flag=None, mask=None):
        self.wl = wl
        self.f_obs = f_obs
        self.f_err = f_err
        self.f_flag = f_flag
        self.mask = mask

        self.i_f_obs = None
        self.i_f_err = None
        self.i_f_flag = None


    def updateIntegratedSpec(self):
        self.i_f_obs, self.i_f_err, self.i_f_flag = integrated_spec(self.f_obs,
                                                                    self.f_err,
                                                                    self.f_flag,
                                                                    self.mask)
    

    def getQbickPlanes(self):
        wl_mask = np.where((self.wl > 5590.0) & (self.wl < 5680.0))[0]
        f_obs = np.ma.masked_invalid(self.f_obs[wl_mask], copy=True)
        f_obs.fill_value = 0.0
        f_obs = f_obs.filled()
        f_flag = self.f_flag[wl_mask]
        wl = self.wl[wl_mask]
        
        planes = np.zeros(shape=f_obs.shape[1:],
                          dtype=[
                                 ('Signal', 'float64'),
                                 ('Noise', 'float64'),
                                 ('Sn', 'float64'),
                                 ('ZonesNoise', 'float64'),
                                 ('ZonesSn', 'float64'),
                                 ])
        
        snout = calc_sn(wl, f_obs[:, self.mask], f_flag[:, self.mask])
        planes['Signal'][self.mask] = snout[0]
        planes['Noise'][self.mask] = snout[1]
        planes['ZonesNoise'][self.mask] = snout[1]
        planes['Sn'][self.mask] = snout[2]
        planes['ZonesSn'][self.mask] = snout[2]
        return  planes
    
    
    def writeHDF5(self, db, parent, name, overwrite=False):
        self.updateIntegratedSpec()
        
        if overwrite and name in parent:
            logger.warn('Removing existing array %s' % name)
            parent._f_getChild(name)._f_remove(recursive=True)
        grp = db.createGroup(parent, name)
        
        save_array(db, grp, 'f_obs', self.f_obs, overwrite)
        save_array(db, grp, 'f_err', self.f_err, overwrite)
        save_array(db, grp, 'f_flag', self.f_flag, overwrite)
        save_array(db, grp, 'mask', self.mask, overwrite)
        save_array(db, grp, 'i_f_obs', self.i_f_obs, overwrite)
        save_array(db, grp, 'i_f_err', self.i_f_err, overwrite)
        save_array(db, grp, 'i_f_flag', self.i_f_flag, overwrite)
        
        
    def loadHDF5(self):
        raise NotImplementedError()
    
################################################################################


################################################################################
class DecompContainer(object):
    def __init__(self):
        self.total = IFSContainer()
        self.bulge = IFSContainer()
        self.disk = IFSContainer()
        self.mask = None
        self.zones = None
        self.firstPassParams = None
        self.fitParams = None
        self.attrs = {}
        
        
    def updateIntegratedSpec(self):
        self.total.updateIntegratedSpec()
        self.bulge.updateIntegratedSpec()
        self.disk.updateIntegratedSpec()
    

    def updateErrorsFlags(self, flag_bad_fit):
        self._computeComponentErrorsFlags(self.bulge, flag_bad_fit)
        self._computeComponentErrorsFlags(self.disk, flag_bad_fit)
    
    
    def _computeComponentErrorsFlags(self, comp, flag_bad_fit):
        shape = self.total.f_obs.shape
        comp.f_err = np.zeros(shape=shape)
        r_comp = np.zeros(shape=shape)
        good = self.total.f_obs > 0
        r_comp[good] = comp.f_obs[good] / self.total.f_obs[good]
        comp.f_err[good] = np.sqrt(r_comp[good]) * self.total.f_err[good]
        comp.f_flag = self.total.f_flag.copy()
        comp.f_flag += flag_big_error(comp.f_obs, comp.f_err)
        comp.f_flag += flag_small_error(comp.f_obs, comp.f_err, comp.f_flag)
        comp.f_flag += flag_bad_fit


    def writeHDF5(self, db_file, sampleId, galaxyId, overwrite=False):
        with open_file(db_file, 'a') as db:
            galaxyNode = califa_id_str(galaxyId)
            try:
                grp = db.getNode('/%s/%s' % (sampleId, galaxyNode))
            except:
                grp = db.createGroup('/%s' % sampleId, galaxyNode, createparents=True)
    
            if overwrite and 'first_pass_parameters' in grp:
                grp.first_pass_parameters._f_remove()
            if overwrite and 'fit_parameters' in grp:
                grp.fit_parameters._f_remove()
                
            self._writeHDF5Tables(db, grp)
            self.total.writeHDF5(db, grp, 'total', overwrite)
            self.bulge.writeHDF5(db, grp, 'bulge', overwrite)
            self.disk.writeHDF5(db, grp, 'disk', overwrite)
            save_array(db, grp, 'zones', self.zones, overwrite)
        
        
    def _writeHDF5Tables(self, db, grp):
        t = db.createTable(grp, 'fit_parameters', self.fitParams.dtype,
                           'Morphology fit parameters', Filters(1, 'blosc'),
                           expectedrows=len(self.fitParams))
        for k, val in self.attrs.iteritems():
            t.attrs[k] = val
        t.append(self.fitParams)
        t.flush()
        
        t = db.createTable(grp, 'first_pass_parameters', self.firstPassParams.dtype,
                           'Morphology first pass fit parameters', Filters(1, 'blosc'),
                           expectedrows=len(self.firstPassParams))
        t.append(self.firstPassParams)
        t.flush()


    def loadHDF5(self):
        raise NotImplementedError()
################################################################################


################################################################################
def save_compound_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        logger.warn('Removing existing group %s' % name)
        parent._f_getChild(name)._f_remove(recursive=True)
    grp = db.createGroup(parent, name)
    # HACK: pytables does not support compound dtypes.
    for field in data.dtype.names:
        save_array(db, grp, field, data[field], overwrite)
################################################################################

    
################################################################################
def save_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        logger.warn('Removing existing array %s' % name)
        parent._f_getChild(name)._f_remove()
    ca = db.createCArray(parent, name, Atom.from_dtype(data.dtype), data.shape,
                         filters=Filters(1, 'blosc'))
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled()
    ca[...] = data
################################################################################
