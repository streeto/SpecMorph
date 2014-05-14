'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
from pycasso.util import logger
from imfit import Imfit, moffat_psf, gaussian_psf

import collections
import time
from os import path, unlink
import numpy as np

from .model import GalaxyModel


__all__ = ['BulgeDiskDecomposition']
FWHM_to_sigma_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))


################################################################################
class BulgeDiskDecomposition(fitsQ3DataCube):
    minNPix = 1000
    vdPercentile = 95.0

    def __init__(self, synthesisFile, smooth=True, use_fobs=True, target_vd=None,
                 PSF_FWHM=0.0, PSF_beta=-1, PSF_size=15, purge_cache=False, nproc=-1):
        self._nproc = nproc
        self._useFobs = use_fobs
        fitsQ3DataCube.__init__(self, synthesisFile, smooth)
        self._synthesisFile = synthesisFile
        self._calcRestFrameSpectra(target_vd)
        self.f_syn_rest__lyx = self.zoneToYX(self.f_syn_rest__lz, extensive=True, surface_density=False)
        self.f_obs_rest__lyx = self.zoneToYX(self.f_obs_rest__lz, extensive=True, surface_density=False)
        self.f_err_rest__lyx = self.zoneToYX(self.f_err_rest__lz, extensive=True, surface_density=False)
        self.f_flag_rest__lyx = self.zoneToYX(self.f_flag_rest__lz, extensive=False)
        if PSF_beta > 0.0:
            self._PSF = moffat_psf(PSF_FWHM, PSF_beta, size=PSF_size)
        else:
            self._PSF = gaussian_psf(PSF_FWHM, size=PSF_size)
    
    
    def _calcRestFrameSpectra(self, target_vd):
        if target_vd is None:
            target_vd = np.percentile(self.v_d, self.vdPercentile)
        self.target_vd = target_vd
        logger.info('Target v_d = %f' % self.target_vd)
        
        from pystarlight.util.velocity_fix import SpectraVelocityFixer
        fix_spectra = SpectraVelocityFixer(self.l_obs, self.v_0, self.v_d, self._nproc)
        logger.debug('Computing rest frame spectra from scratch...')
        self.f_syn_rest__lz = fix_spectra(self.f_syn, target_vd)
        self.f_obs_rest__lz = fix_spectra(self.f_obs, target_vd)
        self.f_err_rest__lz = fix_spectra(self.f_err, target_vd)
        # flag anything that touched a flagged pixel.
        self.f_flag_rest__lz = np.where(fix_spectra(self.f_flag, target_vd) > 0.0, 1.0, 0.0)
    
    
    def _getSpectraSlice(self, l1, l2=None, flag_ratio_threshold=0.5):
        if l1 < 0:
            l1 = 0
        if l2 >= self.Nl_obs:
            l2 = self.Nl_obs - 1

        if (l1 == l2) or (l2 is None):
            wl = self.l_obs[l1]
            flag = self.f_flag_rest__lyx[l1] > 0
            if self._useFobs:
                f = self.f_obs_rest__lyx[l1] / self.flux_unit
            else:
                f = self.f_syn_rest__lyx[l1] / self.flux_unit
            noise = self.f_err_rest__lyx[l1] / self.flux_unit
        else:
            wl = np.mean(self.l_obs[l1:l2])
            flag__l = self.f_flag_rest__lyx[l1:l2] > 0
            n_lambda = flag__l.shape[0]
            flag = flag__l.sum(axis=0) > (flag_ratio_threshold * n_lambda)
            if self._useFobs:
                f = self.f_obs_rest__lyx[l1:l2] / self.flux_unit
            else:
                f = self.f_syn_rest__lyx[l1:l2] / self.flux_unit
            f[flag__l] = np.ma.masked
            f = np.median(f, axis=0)
            noise = self.f_err_rest__lyx[l1:l2] / self.flux_unit
            noise[flag__l] = np.ma.masked
            noise = 1.0/np.sqrt(np.sum(noise**-2, axis=0))
            
        # HACK: Valid noise should not be zero, but some spectra forget about it.
        flag |= (noise == 0.0)
        f[flag] = np.ma.masked
        noise[flag] = np.ma.masked

        return f, noise, wl


    def specSlicer(self, step, box_radius):
        for wl_ix in np.arange(0, self.Nl_obs, step):
            yield self._getSpectraSlice(wl_ix - box_radius, wl_ix + box_radius)
    
    
    def _guessInitialModel(self):
        model_file = self._synthesisFile + '.initmodel'
        if path.exists(model_file):
            try:
                guess_model = GalaxyModel.readConfig(model_file)
                logger.debug('Initial model found:\n%s\n' % guess_model)
                return guess_model
            except:
                logger.warn('Bad model file %s. Deleting.' % model_file)
                unlink(model_file)
        
        logger.warn('Computing initial model using DE algorithm (takes a LOT of time).')
        t1 = time.time()
        mask = ~self.qMask
        qSignal = np.ma.array(self.qSignal, mask=mask)
        qNoise = np.ma.array(self.qNoise, mask=mask)

        pa, ba = self.getEllipseParams()
        ell = 1 - ba
        pa = pa * 180.0 / np.pi
        if pa < 0.0:
            pa += 180.0
        guess_model = GalaxyModel(wl=5635.0, x0=self.x0, y0=self.y0,
                                  I_e=qSignal.max(), r_e=self.HLR_pix/2.0, n=3, PA_b=pa, ell_b=ell,
                                  I_0=qSignal.max(), h=self.HLR_pix/2.0, PA_d=pa, ell_d=ell)
        logger.debug('Initial model:\n%s\n' % guess_model)
        # Fit qSignal to find the first guess.
        imfit = Imfit(guess_model, self._PSF, quiet=False, nproc=self._nproc)
        imfit.fit(qSignal, qNoise, mode='DE')
        guess_model = imfit.getModelDescription()
        logger.debug('Refined initial model:\n%s\n' % guess_model)
        logger.warn('Initial model time: %.2f\n' % (time.time() - t1))
        with open(model_file, 'w') as f:
            logger.debug('Saving model config %s.' % model_file)
            f.write(str(guess_model))
        return guess_model


    def _getInitialModelIterator(self, initial_model, step):
        if initial_model is None:
            initial_model = self._guessInitialModel()

        if isinstance(initial_model, collections.Iterable):
            return iter(initial_model[::step])
        else:
            def constant_model(model):
                for _ in xrange(self.getNSlices(step)):
                    yield model
            return constant_model(initial_model)
    
    
    def getNSlices(self, step):
        n = int(self.Nl_obs / step)
        if self.Nl_obs % step > 0:
            n += 1
        return n
    
    
    def _fit(self, flux, noise, model, mode='LM', insist=False):
        N_pix = (~flux.mask).sum()
        imfit = Imfit(model, self._PSF, quiet=True, nproc=self._nproc)
        if N_pix > self.minNPix:
            imfit.fit(flux, noise, mode=mode)
            logger.debug('Valid pix: %d | Iterations: %d | pegged: %d | chi2: %f' % \
                         (imfit.nValidPixels, imfit.nIter, imfit.nPegged, imfit.chi2))
            fitted_model = imfit.getModelDescription()
            if not imfit.fitConverged or imfit.nPegged > 0:
                logger.warn('Bad fit: did not converge or pegged parameter.')
                fitted_model.flag = 1.0
                if mode == 'LM' and insist:
                    logger.warn('     Retrying using N-M simplex.')
                    imfit.fit(flux, noise, mode='NM')
                    fitted_model = imfit.getModelDescription()
                    if imfit.fitConverged:
                        logger.debug('     N-M simplex chi2: %f' % imfit.chi2)
                    else:
                        logger.warn('     Bad fit: N-M simplex did not converge.')
                        logger.debug('     Initial model:\n%s\n\n' % str(model))
                        fitted_model.flag = 0.0
            fitted_model.chi2 = imfit.chi2
            fitted_model.nValidPixels = imfit.nValidPixels
        else:
            logger.warn('Bad fit: not enough pixels (%d).' % N_pix)
            fitted_model = imfit.getModelDescription()
            fitted_model.flag = 1.0
            fitted_model.chi2 = np.nan
            fitted_model.nValidPixels = N_pix
        return fitted_model


    def fitSpectra(self, step=1, box_radius=0, initial_model=None, mode='LM', insist=False):
        initial_model = self._getInitialModelIterator(initial_model, step)
        slices = self.specSlicer(step, box_radius)
        models = []
        for flux, noise, wl in slices:
            logger.debug('Fitting for wavelength: %.0f \\AA' % wl)
            fitted_model = self._fit(flux, noise, initial_model.next(), mode, insist)
            fitted_model.wl = wl
            models.append(fitted_model)
                
        return models
    
    
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

