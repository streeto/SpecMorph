'''
Created on 20/06/2014

@author: andre
'''

from ..decomposition import IFSDecomposer
from ..util import logger
from ..kinematics import fix_kinematics

from pycasso.fitsdatacube import fitsQ3DataCube
import numpy as np
import pyfits

__all__ = ['CALIFADecomposer', 'save_qbick_images']

################################################################################
class CALIFADecomposer(IFSDecomposer):
    vdPercentile = 95.0
    dispFWHM_V500 = 6.0  # AA
    dispFWHM_V1200 = 2.3 # AA
    
    def __init__(self, db, target_vd=None, use_fobs=True, grating='none', nproc=None):
        IFSDecomposer.__init__(self)
        if grating == 'V500':
            disp_FWHM = self.dispFWHM_V500
        elif grating == 'V1200':
            disp_FWHM = self.dispFWHM_V1200
        elif grating == 'none':
            disp_FWHM = None
        self.K = fitsQ3DataCube(db, smooth=True)
        flux, error, flags, vel_FWHM = self._fixKinematics(target_vd, disp_FWHM, nproc)
        if disp_FWHM is not None:
            wl_FWHM = np.sqrt(vel_FWHM**2 + disp_FWHM**2)
        else:
            wl_FWHM = None
        logger.debug('Computing IFS from voronoi zone spectra...')
        flux = self.K.zoneToYX(flux, extensive=True, surface_density=False)
        error = self.K.zoneToYX(error, extensive=True, surface_density=False)
        flags = self.K.zoneToYX(flags, extensive=False)
        self.loadData(self.K.l_obs, flux, error, flags, wl_FWHM)


    def _fixKinematics(self, target_vd, wl_FWHM=None, nproc=None):
        flags = self.K.f_flag > 0.0
        if target_vd is None:
            target_vd = np.percentile(self.K.v_d, self.vdPercentile)
        self.targetVd = target_vd
        c = 2.997925e5
        lambda_zero = 5500.0
        dl = lambda_zero * target_vd / c
        logger.debug('Fixing kinematics: v_d = %.1f km/s (%.1f \\AA @ 5500 \\AA) ...' % (target_vd, dl))
        flux, error, flags = fix_kinematics(self.K.l_obs, self.K.f_obs, self.K.f_err, flags,
                                            self.K.v_0, self.K.v_d, target_vd,
                                            nproc, wl_FWHM)
        return flux, error, flags, dl
    

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
    
    
    def getUpdatedQbickHDU(self, new_planes):
        phdu = self.K.getPrimaryHdu().copy()
        for pname in new_planes.dtype.names:
            planeId = self.K._planeIndex[pname]
            phdu.data[planeId] = new_planes[pname]
        
        # TODO: remove all pycasso headers
        for key in phdu.header.keys():
            if key.startswith('SYN'):
                phdu.header.remove(key)
                
        return phdu

################################################################################


################################################################################
def save_qbick_images(comp, decomp, filename, overwrite=False):    
    new_planes = comp.getQbickPlanes()
    phdu = decomp.getUpdatedQbickHDU(new_planes)
    hdulist = pyfits.HDUList()
    hdulist.append(phdu)
    hdulist.writeto(filename, clobber=overwrite)
################################################################################
