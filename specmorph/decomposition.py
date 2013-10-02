'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
from pycasso.util import logger
from imfit.fitting import Imfit
from imfit.psf import moffat_psf, gaussian_psf

import numpy as np
from os import path, unlink

from .velocity_fix import SpectraVelocityFixer
from .model import GalaxyModel

try:
    import joblib  # @UnusedImport
    joblib_available = True
except:
    joblib_available = False
    
enable_parallel = joblib_available and not __debug__

__all__ = ['BulgeDiskDecomposition']
FWHM_to_sigma_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))


################################################################################
class BulgeDiskDecomposition(fitsQ3DataCube):

    def __init__(self, synthesisFile, smooth=True, target_vd=0.0,
                 PSF_FWHM=0.0, PSF_beta=None, PSF_size=15, purge_cache=False, nproc=-1):
        self._nproc = nproc
        fitsQ3DataCube.__init__(self, synthesisFile, smooth)
        self._loadRestFrameSpectra(synthesisFile + '.rest-spectra.h5', target_vd, purge_cache)
        self.f_syn_rest__lyx = self.zoneToYX(self.f_syn_rest__lz, extensive=True, surface_density=False)
        self.f_obs_rest__lyx = self.zoneToYX(self.f_obs_rest__lz, extensive=True, surface_density=False)
        self.f_err_rest__lyx = self.zoneToYX(self.f_err_rest__lz, extensive=True, surface_density=False)
        self.f_flag_rest__lyx = self.zoneToYX(self.f_flag_rest__lz, extensive=False)
        if PSF_beta is None:
            self._PSF = gaussian_psf(PSF_FWHM, size=PSF_size)
        else:
            self._PSF = moffat_psf(PSF_FWHM, PSF_beta, size=PSF_size)
    
    
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
    
    
    def _getSpectraSlice(self, l1, l2=None, flag_ratio_threshold=0.5):
        if l1 < 0:
            l1 = 0
        if l2 >= self.Nl_obs:
            l2 = self.Nl_obs - 1
        if (l1 == l2) or (l2 is None):
            wl = self.l_obs[l1]
            flag = self.f_flag_rest__lyx[l1] > 0
            
            f = self.f_syn_rest__lyx[l1] / self.flux_unit
            f[flag] = np.ma.masked 
            f[flag] = 0.0
            f.fill_value = 0.0 
            
            noise = self.f_err_rest__lyx[l1] / self.flux_unit
            noise[flag] = np.ma.masked
            noise_fill = noise.max()
            noise[flag] = noise_fill
            noise.fill_value = noise_fill
        else:
            wl = np.mean(self.l_obs[l1:l2])
            flag = self.f_flag_rest__lyx[l1:l2] > 0
            n_lambda = flag.shape[0]
            flag__yx = flag.sum(axis=0) > (flag_ratio_threshold * n_lambda)
            n_good_pix = (~flag__yx).sum()
            
            f = self.f_syn_rest__lyx[l1:l2] / self.flux_unit
            f[flag] = np.ma.masked
            f = np.median(f, axis=0)
            f[flag__yx] = np.ma.masked
            f.fill_value = 0.0

            noise = self.f_err_rest__lyx[l1:l2] / self.flux_unit
            noise[flag] = np.ma.masked
            noise = 1.0/np.sqrt(np.sum(noise**-2, axis=0))
            noise[flag__yx] = np.ma.masked
            if n_good_pix > 0:
                noise_fill = noise.max()
            else:
                noise_fill = 1.0
            noise.fill_value = noise_fill

        return f, noise, wl


    def specSlicer(self, step, box_radius):
        for wl_ix in np.arange(0, self.Nl_obs, step):
            yield self._getSpectraSlice(wl_ix - box_radius, wl_ix + box_radius)
    
    
    def fitSpectra(self, step=1, box_radius=0):
        guess_model = GalaxyModel(x0=self.x0, y0=self.y0,
                                  I_e=0.06, r_e=11.0, PA_b=90.0, ell_b=0.35,
                                  I_0=0.1, h=12.0, PA_d=90.0, ell_d=0.3)
        logger.debug('Initial model:\n%s\n' % guess_model)
        # Fit qSignal to find the first guess.
        imfit = Imfit(guess_model, self._PSF, quiet=True, nproc=self._nproc)
        mask = ~self.qMask
        qSignal = np.ma.array(self.qSignal, mask=mask)
        qNoise = np.ma.array(self.qNoise, mask=mask)
        imfit.fit(qSignal, qNoise)
        guess_model = imfit.getModelDescription()
        logger.debug('refined initial model:\n%s\n' % imfit.getModelDescription())
        
        slices = self.specSlicer(step, box_radius)
        models = []
        for flux, noise, wl in slices:
            logger.debug('Fitting for wavelength: %.0f \\AA' % wl)
            imfit = Imfit(guess_model, self._PSF, quiet=True, nproc=self._nproc)
            imfit.fit(flux, noise)
            logger.debug('Valid pix: %d | Iterations: %d | chi2: %f' % (imfit.nValidPixels, imfit.nIter, imfit.chi2))
            fitted_model = imfit.getModelDescription()
            if not imfit.fitConverged or imfit.nPegged > 0 or imfit.nValidPixels < 1000:
                logger.warn('Bad fit for wavelength: %.0f \\AA.' % wl)
                fitted_model.flag = 1.0
            fitted_model.chi2 = imfit.chi2
            fitted_model.nValidPixels = imfit.nValidPixels
            models.append(fitted_model)
                
        selected_wl_ix = np.arange(0, self.Nl_obs, step)
        return models, selected_wl_ix
    
    
    def _getModelImage(self, model, PSF):
        imfit = Imfit(model, PSF, quiet=True, nproc=self._nproc)
        shape = (self.N_y, self.N_x)
        return imfit.getModelImage(shape) * self.flux_unit
    
    
    def getModelSpectra(self, models, mask=None):
        if mask is None:
            mask = self.qMask
        bulge = np.empty((len(models), self.N_y, self.N_x))
        disk = np.empty((len(models), self.N_y, self.N_x))
        for i, model in enumerate(models):
            bulge[i] = self._getModelImage(model.getBulge(), self._PSF)
            disk[i] = self._getModelImage(model.getDisk(), self._PSF)
        return bulge, disk
        
        
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

