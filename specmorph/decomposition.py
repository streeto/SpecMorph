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
from .velocity_fix import SpectraVelocityFixer

__all__ = ['BulgeDiskDecomposition']
FWHM_to_sigma_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))


################################################################################
class MorphologyFitWrapper(object):
    
    def __init__(self, x0, y0, sigma, rad_clip_in, rad_clip_out, mode, plot=False):
        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma
        self.rad_clip_in = rad_clip_in
        self.rad_clip_out = rad_clip_out
        self.mode = mode
        self.plot = plot

    def __call__(self, fl__yx):
        fitter = BulgeDiskFitter(fl__yx)
        fitter.setup(x0=self.x0, y0=self.y0, sigma=self.sigma,
                     rad_clip_in=self.rad_clip_in, rad_clip_out=self.rad_clip_out)
        fitter.fitModel(self.mode)
        if self.plot:
            fitter.plot_model(self.mode, interactive=False)
        return fitter.getFitParams()
################################################################################


################################################################################
class BulgeDiskDecomposition(fitsQ3DataCube):

    def __init__(self, synthesisFile, smooth=True, target_vd=0.0, FWHM=0.0, purge_cache=False, nproc=-1):
        self._nproc = nproc
        fitsQ3DataCube.__init__(self, synthesisFile, smooth)
        self._loadRestFrameSpectra(synthesisFile + '.rest-spectra.h5', target_vd, purge_cache)
        self.f_syn_rest__lyx = self.zoneToYX(self.f_syn_rest__lz, extensive=True, surface_density=False)
        self.f_obs_rest__lyx = self.zoneToYX(self.f_obs_rest__lz, extensive=True, surface_density=False)
        self._sigma = FWHM / FWHM_to_sigma_factor
    
    
    def _loadRestFrameSpectra(self, filename, target_vd, purge_cache=False):
        try:
            from tables import openFile, Filters
            from tables import Float64Atom  # @UnresolvedImport
        except:
            logger.warn('Pytables not installed, rest frame spectra cache disabled.')
            self._calcRestFrameSpectra(target_vd)
            return

        if path.exists(filename):
            if purge_cache:
                logger.debug('Forced purge of rest frame spectra cache.')
                unlink(filename)
            else:            
                f = openFile(filename, 'r')
                try:
                    self.f_syn_rest__lz = f.root.f_syn_rest__lz[...]
                    self.f_obs_rest__lz = f.root.f_obs_rest__lz[...]
                    self.f_err_rest__lz = f.root.f_err_rest__lz[...]
                    self.f_flag_rest__lz = f.root.f_flag_rest__lz[...]
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

        self._calcRestFrameSpectra(target_vd)
        f = openFile(filename, 'w')
        try:
            ca = f.createCArray('/', 'f_syn_rest__lz', Float64Atom(),
                                self.f_syn_rest__lz.shape, filters=Filters(1, 'blosc'))
            ca[...] = self.f_syn_rest__lz
            ca = f.createCArray('/', 'f_obs_rest__lz', Float64Atom(),
                                self.f_obs_rest__lz.shape, filters=Filters(1, 'blosc'))
            ca[...] = self.f_obs_rest__lz
            ca = f.createCArray('/', 'f_err_rest__lz', Float64Atom(),
                                self.f_err_rest__lz.shape, filters=Filters(1, 'blosc'))
            ca[...] = self.f_err_rest__lz
            ca = f.createCArray('/', 'f_flag_rest__lz', Float64Atom(),
                                self.f_flag_rest__lz.shape, filters=Filters(1, 'blosc'))
            ca[...] = self.f_flag_rest__lz
            
        finally:
            f.close()


    def _calcRestFrameSpectra(self, target_vd):
        fix_spectra = SpectraVelocityFixer(self.l_obs, self.v_0, self.v_d, self._nproc)
        logger.warn('Computing rest frame spectra from scratch... (go take a coffee)')
        self.f_syn_rest__lz = fix_spectra(self.f_syn, target_vd)
        self.f_obs_rest__lz = fix_spectra(self.f_obs, target_vd)
        self.f_err_rest__lz = fix_spectra(self.f_err, target_vd)
        # flag anything that touched a flagged pixel.
        self.f_flag_rest__lz = np.where(fix_spectra(self.f_flag, target_vd) > 0.0, 1.0, 0.0)
        logger.warn('Done.')
    
    
    def _getSpectraSlice(self, l1, l2=None):
        if l1 < 0:
            l1 = 0
        if l2 >= self.Nl_obs:
            l2 = self.Nl_obs - 1
        if (l1 == l2) or (l2 is None):
            return self.f_syn_rest__lyx[l1] / self.flux_unit
        else:
            nl = l2 - l1
            return self.f_syn_rest__lyx[l1:l2].sum(axis=0) / nl / self.flux_unit


    def specSlicer(self, step, box_radius):
        for wl_ix in np.arange(0, self.Nl_obs, step):
            if __debug__:
                wl = self.l_obs[wl_ix]
                logger.debug('Modeling for lambda = %d \AA' % wl)
            yield self._getSpectraSlice(wl_ix - box_radius, wl_ix + box_radius)
    
    
    def fitSpectra(self, step=1, box_radius=0, rad_clip_in=2.5, rad_clip_out=None, mode='scatter'):
        plot = __debug__ and step > 400
        morphology_fit = MorphologyFitWrapper(self.x0, self.y0, self._sigma, rad_clip_in, rad_clip_out, mode, plot=plot)
        spec_slices = self.specSlicer(step, box_radius)
        try:
            if plot: raise Exception('Plotting in parallel mode is no allowed, faking error.')
            from joblib import Parallel, delayed
            fit_params = Parallel(n_jobs=self._nproc)(delayed(morphology_fit)(fl__yx) for fl__yx in spec_slices)
        except ImportError:
            logger.warn('joblib not installed, falling back to serial processing.')
            fit_params = [morphology_fit(fl__yx) for fl__yx in spec_slices]
        fit_params = np.array(fit_params, dtype=BulgeDiskFitter.getParamDtype())
        selected_wl_ix = np.arange(0, self.Nl_obs, step)
        return fit_params, selected_wl_ix
    
    
    def _getModelImage(self, model, x0, y0, pa, ba, mask=None):
        if mask is None:
            mask = self.qMask
        shape = (self.N_y, self.N_x)
        r__yx = getImageDistance(shape, x0, y0, pa, ba)
        image = np.ma.masked_where(~mask, model(r__yx))
        image.fill_value = self.fill_value
        return image

        
    def getModelSpectra(self, params, mask=None):
        bulge_spectra = np.empty((len(params), self.N_y, self.N_x))
        disk_spectra = np.empty((len(params), self.N_y, self.N_x))
        for i, p in enumerate(params):
            bulge_model = BulgeModel2D(p['I_Be'], p['R_e'], self._sigma)
            disk_model = DiskModel2D(p['I_D0'], p['R_0'], self._sigma)
            x0 = p['x0']
            y0 = p['y0']
            pa = p['pa']
            ba = p['ba']

            bulge_spectra[i] = self._getModelImage(bulge_model, x0, y0, pa, ba, mask).filled() * self.flux_unit
            disk_spectra[i] = self._getModelImage(disk_model, x0, y0, pa, ba, mask).filled() * self.flux_unit
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

