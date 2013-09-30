'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
from pycasso.util import logger
from imfit.fitting import Imfit
from imfit.psf import moffat_psf

import numpy as np
from os import path, unlink
from copy import deepcopy

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

    def __init__(self, synthesisFile, smooth=True, target_vd=0.0, FWHM=0.0, purge_cache=False, nproc=-1):
        self._nproc = nproc
        fitsQ3DataCube.__init__(self, synthesisFile, smooth)
        self._loadRestFrameSpectra(synthesisFile + '.rest-spectra.h5', target_vd, purge_cache)
        self.f_syn_rest__lyx = self.zoneToYX(self.f_syn_rest__lz, extensive=True, surface_density=False)
        self.f_obs_rest__lyx = self.zoneToYX(self.f_obs_rest__lz, extensive=True, surface_density=False)
        self.f_err_rest__lyx = self.zoneToYX(self.f_err_rest__lz, extensive=True, surface_density=False)
        self.f_flag_rest__lyx = self.zoneToYX(self.f_flag_rest__lz, extensive=False)
        self._FWHM = FWHM
    
    
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
            f = self.f_syn_rest__lyx[l1] / self.flux_unit
            noise = self.f_err_rest__lyx[l1] / self.flux_unit
            flag = self.f_flag_rest__lyx[l1] > 0
            f[flag] = np.ma.masked
            mask = f.mask.copy()
            f.fill_value = 0.0 
            f = f.filled()
            noise[flag] = np.ma.masked
            noise.fill_value = noise.max()
            noise = noise.filled()
            wl = self.l_obs[l1]
        else:
            raise NotImplementedError('Fat slices not supported yet.')
        
        return np.ma.array(f, mask=mask), np.ma.array(noise, mask=mask), wl


    def specSlicer(self, step, box_radius):
        for wl_ix in np.arange(0, self.Nl_obs, step):
            yield self._getSpectraSlice(wl_ix - box_radius, wl_ix + box_radius)
    
    
    def fitSpectra(self, step=1, box_radius=0):
        PSF = moffat_psf(self._FWHM, size=51)

        model = GalaxyModel(x0=self.x0, y0=self.y0,
                            I_e=1, r_e=25, PA_b=90.0, ell_b=0.5,
                            I_0=1, h=25, PA_d=90.0, ell_d=0.5)
        imfit = Imfit(model, PSF)
        # Fit qSignal to find the first guess.
        mask = ~self.qMask
        qSignal = np.ma.array(self.qSignal, mask=mask)
        qNoise = np.ma.array(self.qNoise, mask=mask)
        imfit.fit(qSignal, qNoise, quiet=True)
        guess_model = imfit.getModelDescription()
        
        logger.debug('qSignal model: %s' % imfit.getModelDescription())
        slices = self.specSlicer(step, box_radius)
        models = []
        for flux, noise, wl in slices:
            logger.debug('Fitting for wavelength: %.0f \\AA' % wl)
            # FIXME: get rid of explicit deepcopying.
            imfit = Imfit(deepcopy(guess_model), PSF, nproc=self._nproc)
            imfit.fit(flux, noise, quiet=True)
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
        imfit = Imfit(model, PSF)
        shape = (self.N_y, self.N_x)
        return imfit.getModelImage(shape) * self.flux_unit
    
    
    def getModelSpectra(self, models, mask=None):
        if mask is None:
            mask = self.qMask
        PSF = moffat_psf(self._FWHM, size=51) if self._FWHM is not None else None
        bulge = np.empty((len(models), self.N_y, self.N_x))
        disk = np.empty((len(models), self.N_y, self.N_x))
        for i, model in enumerate(models):
            bulge[i] = self._getModelImage(model.getBulge(), PSF)
            disk[i] = self._getModelImage(model.getDisk(), PSF)
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

