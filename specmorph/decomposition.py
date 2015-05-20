'''
Created on Jun 6, 2013

@author: andre
'''

from .util import logger
from .fitting import fit_image, model_image

from imfit import moffat_psf, gaussian_psf
import collections
import numpy as np
from copy import deepcopy
from specmorph.util import gaussian_covariance

__all__ = ['IFSDecomposer']


################################################################################
class IFSDecomposer(object):
    minNPix = 2000
    useEstimatedVariance = False

    
    def __init__(self):
        self._PSF = None


    def loadData(self, wl, flux, error, flags, wl_FWHM=None):
        self.wl = wl
        self.flux = flux
        self.error = error
        self.flags = flags
        self.Nl_obs, self.N_y, self.N_x = flux.shape
        self.wlFWHM = wl_FWHM
        
        
    def setSynthPSF(self, FWHM=0.0, beta=None, size=15):
        if beta is not None:
            self.setPSF(moffat_psf(FWHM, beta, size=size))
        else:
            self.setPSF(gaussian_psf(FWHM, size=size))

    
    def setPSF(self, PSF):
        self._PSF = np.asanyarray(PSF)


    @property
    def PSF(self):
        return self._PSF
    
    
    def getSpectraSlice(self, l1, l2=None, masked_wl=None, flag_ratio_threshold=0.5):
        if l1 < 0:
            l1 = 0
        if l2 >= self.Nl_obs:
            l2 = self.Nl_obs - 1
        if masked_wl is None:
            masked_wl = np.zeros(self.wl.shape, dtype='bool')
        if (l1 == l2) or (l2 is None):
            wl = self.wl[l1]
            flag = self.flags[l1] | masked_wl[l1]
            f_mean = self.flux[l1].copy()
            n_mean = self.error[l1].copy()
        else:
            wl = np.mean(self.wl[l1:l2])
            flag_wl = self.flags[l1:l2].copy()
            flag_wl[masked_wl[l1:l2]] = True
            n_lambda = flag_wl.shape[0]
            n_lambda_flagged = flag_wl.sum(axis=0)
            flag = n_lambda_flagged > (flag_ratio_threshold * n_lambda)
            f = self.flux[l1:l2].copy()
            n = self.error[l1:l2].copy()
            f[flag_wl] = np.ma.masked
            n[flag_wl] = np.ma.masked
            w = n**-2
            w_norm = w.sum(axis=0)
            f_mean = (f * w).sum(axis=0) / w_norm
            if self.useEstimatedVariance:
                w2_norm = (w * w).sum(axis=0)
                sigma2 = w_norm / (w_norm**2 - w2_norm) * (w * (f - f_mean)**2).sum(axis=0)
            else:
                sigma2 = w_norm**-1
                if self.wlFWHM is not None:
                    a = gaussian_covariance(n, self.wl[l1:l2], self.wlFWHM)
                    print a.compressed()
                    print sigma2.compressed()
                    sigma2 += a
            n_mean = np.sqrt(sigma2)

        f_mean[flag] = np.ma.masked
        n_mean[flag] = np.ma.masked

        validpix = (~flag).sum()
        assert validpix == 0 or np.isfinite(f_mean).all()
        assert validpix == 0 or np.isfinite(n_mean).all()
        assert validpix == 0 or (f_mean > 0.0).all()
        assert validpix == 0 or (n_mean > 0.0).all()
        
        return f_mean, n_mean, wl


    def _specSlicer(self, step, box_radius, masked_wl):
        for wl_ix in np.arange(0, self.Nl_obs, step):
            yield self.getSpectraSlice(wl_ix - box_radius, wl_ix + box_radius, masked_wl)
    
    
    def _getInitialModelIterator(self, initial_model, step):
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
    
    
    def _fit(self, flux, noise, model, mode='LM', insist=False, nproc=None):
        N_pix = (~flux.mask).sum()
        if N_pix > self.minNPix:
            fitted_model, success, chi2 = fit_image(flux, noise, model, self.PSF,
                                                    mode=mode, insist=insist, quiet=True, nproc=nproc)
            fitted_model.chi2 = chi2
            fitted_model.flag = not success
            fitted_model.nValidPixels = N_pix
            
        else:
            logger.warn('Bad fit: not enough pixels (%d).' % N_pix)
            fitted_model = deepcopy(model)
            fitted_model.flag = True
            fitted_model.chi2 = np.nan
            fitted_model.nValidPixels = N_pix
        return fitted_model


    def fitSpectra(self, step=1, box_radius=0, initial_model=None, mode='LM',
                   insist=False, masked_wl=None, nproc=None):
        initial_model = self._getInitialModelIterator(initial_model, step)
        slices = self._specSlicer(step, box_radius, masked_wl=masked_wl)
        models = []
        for flux, noise, wl in slices:
            logger.debug('Fitting for wavelength: %.0f \\AA' % wl)
            fitted_model = self._fit(flux, noise, initial_model.next(), mode, insist, nproc)
            fitted_model.wl = wl
            models.append(fitted_model)
                
        return models
    
    
    def getModelSpectra(self, models, nproc=None, use_PSF=True):
        '''
        FIXME: generic number of components.
        '''
        if use_PSF:
            PSF = self.PSF
        else:
            PSF = None
        bulge = np.empty((len(models), self.N_y, self.N_x))
        disk = np.empty((len(models), self.N_y, self.N_x))
        shape = (self.N_y, self.N_x)
        for i, model in enumerate(models):
            bulge[i] = model_image(model.getBulge(), shape, PSF, nproc)
            disk[i] = model_image(model.getDisk(), shape, PSF, nproc)
        return bulge, disk
################################################################################
