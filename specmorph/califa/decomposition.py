'''
Created on 20/06/2014

@author: andre
'''

from ..decomposition import IFSDecomposer
from ..util import logger
from ..kinematics import fix_kinematics

from pycasso.fitsdatacube import fitsQ3DataCube
import numpy as np


################################################################################
class CALIFADecomposer(IFSDecomposer):
    vdPercentile = 95.0
    
    def __init__(self, db, target_vd=None, use_fobs=True, nproc=None):
        IFSDecomposer.__init__(self)
        self.K = fitsQ3DataCube(db, smooth=True)
        wl = self.K.l_obs
        if use_fobs:
            flux = self.K.f_obs
        else:
            flux = self.K.f_syn
        error = self.K.f_err
        flags = self.K.f_flag > 0.0
        if target_vd is None:
            target_vd = np.percentile(self.K.v_d, self.vdPercentile)
        self.targetVd = target_vd
        logger.debug('Fixing kinematics...')
        flux, error, flags = fix_kinematics(self.K.l_obs, flux, error, flags,
                                            self.K.v_0, self.K.v_d, target_vd, nproc)
        logger.debug('Computing IFS from voronoi zone spectra...')
        flux = self.K.zoneToYX(flux, extensive=True, surface_density=False)
        error = self.K.zoneToYX(error, extensive=True, surface_density=False)
        flags = self.K.zoneToYX(flags, extensive=False)
        self.loadData(wl, flux, error, flags)


    def YXToZone(self, prop, extensive=True, surface_density=True):
        if prop.ndim == 2:
            return self._YXToZone2d(prop, extensive, surface_density)
        else:
            return self._YXToZoneNd(prop, extensive, surface_density)
        
        
    def _YXToZone2d(self, prop, extensive=True, surface_density=True):
        if prop.ndim != 2:
            raise ValueError('prop must be 2-dimensional.')
        prop__z = np.empty(self.K.N_zone, dtype=prop.dtype)
        for _z in xrange(self.K.N_zone):
            _p = prop[self.K.qZones == _z].sum()
            if not extensive:
                _p /= self.K.zoneArea_pix[_z]
            if surface_density:
                _p *= self.K.parsecPerPixel**2
            prop__z[_z] = _p
        return prop__z
    
    def _YXToZoneNd(self, prop, extensive=True, surface_density=True):
        if prop.ndim <= 2:
            raise ValueError('prop must be at least 3-dimensional.')
        shape = prop.shape[:-2] + (self.K.N_zone,)
        prop__z = np.empty(shape, dtype=prop.dtype)
        for _z in xrange(self.K.N_zone):
            _p = prop[...,self.K.qZones == _z].sum(axis=-1)
            if not extensive:
                _p /= self.K.zoneArea_pix[_z]
            if surface_density:
                _p *= self.K.parsecPerPixel**2
            prop__z[...,_z] = _p
        return prop__z
################################################################################
