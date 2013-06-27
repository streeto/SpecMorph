'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
from pycasso.util import logger, getImageDistance
import numpy as np
from os import path, unlink

from .fitting import BulgeDiskFitter
from .model import BulgeModel2D, DiskModel2D
from gauss_smooth import gaussVelocitySmooth  # @UnresolvedImport

__all__ = ['BulgeDiskDecomposition']
FWHM_to_sigma_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))


################################################################################
def morphology_fit(fl__yx, x0, y0, sigma, rad_clip_in, rad_clip_out, fit_psf, fit_mode):
    fitter = BulgeDiskFitter(fl__yx)
    fitter.setup(x0=x0, y0=y0, sigma=sigma, rad_clip_in=rad_clip_in, rad_clip_out=rad_clip_out)
    fitter.fitModel(fit_mode, fit_psf=fit_psf)
    return fitter.getFitParams()
################################################################################


################################################################################
class BulgeDiskDecomposition(fitsQ3DataCube):

    def __init__(self, synthesisFile, smooth=True, target_vd=0.0, purge_cache=False):
        fitsQ3DataCube.__init__(self, synthesisFile, smooth)
        self._loadRestFrameSpectra(synthesisFile + '.rest-spectra.h5', target_vd, purge_cache)
        self.f_syn_fixed__lyx = self.zoneToYX(self.f_syn_fixed, extensive=True, surface_density=False)
    
    
    def _loadRestFrameSpectra(self, filename, target_vd, purge_cache=False):
        try:
            from tables import openFile, Filters
            from tables import Float64Atom  # @UnresolvedImport
        except:
            logger.warn('Pytables not installed, cache disabled. Loading will always take a long time!')
            self.f_syn_fixed = self._getSpectraWithFixedVelocities(self.f_syn, target_vd)

        if path.exists(filename):
            if purge_cache:
                logger.debug('Forced purge of rest frame spectra cache.')
                unlink(filename)
            else:            
                f = openFile(filename, 'r')
                try:
                    self.f_syn_fixed = f.root.f_syn_fixed[...]
                    logger.debug('Rest frame spectra cache loaded successfully.')
                    return
                except:
                    logger.warn('Purging corrupt rest frame spectra cache.')
                    f.close()
                    unlink(filename)
                finally:
                    f.close()
        else:
            logger.warn('Rest frame spectra cache not found.')

        logger.warn('Computing cache from scratch... (go take a coffee)')
        f = openFile(filename, 'w')
        try:
            self.f_syn_fixed = self._getSpectraWithFixedVelocities(self.f_syn, target_vd)
            ca = f.createCArray('/', 'f_syn_fixed', Float64Atom(),
                                self.f_syn_fixed.shape, filters=Filters(1, 'blosc'))
            ca[...] = self.f_syn_fixed
            
        finally:
            f.close()

        
    def _getSpectraWithFixedVelocities(self, f, target_vd):
        # Fix the velocity dispersion only if needed.
        m = self.v_d < target_vd
        vd_fix =np.zeros_like(self.v_d)
        vd_fix[m] = np.sqrt(target_vd**2 - self.v_d[m]**2)
        
        f_fixed = np.empty_like(f)
        for i in xrange(self.N_zone):
            fo = f[:,i]
            f_fixed[:,i] = gaussVelocitySmooth(self.l_obs, fo, -self.v_0[i], vd_fix[i])
        return f_fixed
    

    def getSpectraSlice(self, l1, l2=None):
        if l1 < 0:
            l1 = 0
        if l2 >= self.Nl_obs:
            l2 = self.Nl_obs - 1
        if (l1 == l2) or (l2 is None):
            return self.f_syn_fixed__lyx[l1] / self.flux_unit
        else:
            nl = l2 - l1
            return self.f_syn_fixed__lyx[l1:l2].sum(axis=0) / nl / self.flux_unit


    def spectraSliceProducer(self, step, box_radius):
        for wl_ix in np.arange(0, self.Nl_obs, step):
            if __debug__:
                wl = self.l_obs[wl_ix]
                logger.debug('Modeling for lambda = %d \AA' % wl)
            yield self.getSpectraSlice(wl_ix - box_radius, wl_ix + box_radius)
    
    
    def fitSpectra(self, step=1, box_radius=0, FWHM=0.0, rad_clip_in=2.5, rad_clip_out=None, fit_psf=False, mode='scatter', parallel=True):
        if parallel:
            return self._parallelFitSpectra(step, box_radius, FWHM, rad_clip_in, rad_clip_out, fit_psf, mode)
        else:
            return self._serialFitSpectra(step, box_radius, FWHM, rad_clip_in, rad_clip_out, fit_psf, mode)
            

    def _parallelFitSpectra(self, step=1, box_radius=0, FWHM=0.0, rad_clip_in=2.5, rad_clip_out=None, fit_psf=False, mode='scatter'):
        try:
            from joblib import Parallel, delayed
        except:
            logger.critical('joblib not installed.')
            raise
        sigma = FWHM / FWHM_to_sigma_factor
        fit_params = Parallel(n_jobs=-1)(delayed(morphology_fit)(fl__yx, self.x0, self.y0, sigma, rad_clip_in, rad_clip_out, fit_psf, mode) for fl__yx in self.spectraSliceProducer(step, box_radius))
        fit_params = np.array(fit_params, dtype=BulgeDiskFitter.getParamDtype())
        selected_wl_ix = np.arange(0, self.Nl_obs, step)
        return fit_params, selected_wl_ix
    
    
    def _serialFitSpectra(self, step=1, box_radius=0, FWHM=0.0, rad_clip_in=2.5, rad_clip_out=None, fit_psf=False, mode='scatter'):
        sigma = FWHM / FWHM_to_sigma_factor
        fit_params = []
        for fl__yx in self.spectraSliceProducer(step, box_radius):
            params = morphology_fit(fl__yx, self.x0, self.y0, sigma, rad_clip_in, rad_clip_out, fit_psf, mode)
            fit_params.append(params)
        fit_params = np.array(fit_params, dtype=BulgeDiskFitter.getParamDtype())
        selected_wl_ix = np.arange(0, self.Nl_obs, step)
        return fit_params, selected_wl_ix
    
    
    def _getModelImage(self, model, x0, y0, pa, ba, mask=None):
        shape = (self.N_y, self.N_x)
        r__yx = getImageDistance(shape, x0, y0, pa, ba)
        image = model(r__yx)
        if mask is None:
            mask = self.qMask
        image[~mask] = np.nan
        return image

        
    def getModelSpectra(self, params, mask=None):
        bulge_spectra = np.empty((len(params), self.N_y, self.N_x))
        disk_spectra = np.empty((len(params), self.N_y, self.N_x))
        for i, p in enumerate(params):
            bulge_model = BulgeModel2D(p['I_Be'], p['R_e'], p['sigma'])
            disk_model = DiskModel2D(p['I_D0'], p['R_0'], p['sigma'])
            x0 = p['x0']
            y0 = p['y0']
            pa = p['pa']
            ba = p['ba']

            bulge_spectra[i] = self._getModelImage(bulge_model, x0, y0, pa, ba, mask) * self.flux_unit
            disk_spectra[i] = self._getModelImage(disk_model, x0, y0, pa, ba, mask) * self.flux_unit
        return bulge_spectra, disk_spectra
        
        
    def YXToZone(self, prop, extensive=True, surface_density=True):
        if prop.ndim == 2:
            return self._YXToZone2d(prop, extensive, surface_density)
        else:
            return self._YXToZoneNd(prop, extensive, surface_density)
        
        
    def _YXToZone2d(self, prop, extensive=True, surface_density=True):
        if prop.ndim != 2:
            raise ValueError('prop must be 2-dimensional.')
        prop__z = np.empty(self.N_zone, dtype=prop.dtype)
        for _z in xrange(self.N_zone):
            _p = prop[self.qZones == _z].sum()
            if not extensive:
                _p /= self.zoneArea_pix[_z]
            if surface_density:
                _p *= self.parsecPerPixel**2
            prop__z[_z] = _p
        return prop__z
    
    def _YXToZoneNd(self, prop, extensive=True, surface_density=True):
        if prop.ndim <= 2:
            raise ValueError('prop must be at least 3-dimensional.')
        shape = prop.shape[:-2] + (self.N_zone,)
        prop__z = np.empty(shape, dtype=prop.dtype)
        for _z in xrange(self.N_zone):
            _p = prop[...,self.qZones == _z].sum(axis=-1)
            if not extensive:
                _p /= self.zoneArea_pix[_z]
            if surface_density:
                _p *= self.parsecPerPixel**2
            prop__z[...,_z] = _p
        return prop__z
    
################################################################################

