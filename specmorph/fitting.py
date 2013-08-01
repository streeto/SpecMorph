'''
Created on Jun 10, 2013

@author: andre
'''

from astropy.modeling import fitting
from pycasso.util import getEllipseParams, getImageDistance, radialProfile, logger
import numpy as np
from numpy import ma
from .model import BulgeModel, DiskModel, GalaxyModel, PSF1d_gaussian_convolve

__all__ = ['BulgeDiskFitter']

################################################################################
class BulgeDiskFitter(object):
    _model = None
    _sigma = 1.5
    _pixel_fraction = 0.2
    _param_dtype = np.dtype([('I_Be', 'float64'), ('R_e', 'float64'),
                             ('I_D0', 'float64'), ('R_0', 'float64'),
                             ('R2', 'float64'),
                             ('x0', 'float64'), ('y0', 'float64'),
                             ('pa', 'float64'), ('ba', 'float64')])
    
    f__yx = None
    pixelDistance = None
    min_rad = 2.5
    max_rad = None
    x0 = 0.0
    y0 = 0.0
    pa = 0.0
    ba = 1.0
    R2 = np.nan
    tolerance = 100.0
    enable_bounds = False
        

    @classmethod
    def getNparams(cls):
        return len(cls._param_dtype.fields)
        
        
    @classmethod
    def getParamDtype(cls):
        return cls._param_dtype
        
        
    def __init__(self, f__yx, var_f__yx=None):
        self.f__yx = f__yx
        self.var_f__yx = var_f__yx


    def setup(self, x0=None, y0=None, pa=None, ba=None, sigma=0.0, rad_clip_in=0.0, rad_clip_out=None):
        if x0 is None or y0 is None:
            # FIXME: perform a proper centering
            y0, x0 = np.where(self.f__yx == self.f__yx.max())[0]
        self.x0 = x0
        self.y0 = y0

        if pa is None or ba is None:
            pa, ba = getEllipseParams(self.f__yx, x0, y0, mask=self.mask)
        self.pa = pa
        self.ba = ba
        self.pixelDistance = self._getPixelDistance(x0, y0, pa, ba)

        self._sigma = sigma
        self.min_rad = max(rad_clip_in, self._pixel_fraction) 
        self.max_rad = rad_clip_out
        # FIXME: use circular aperture for radial clip.
        self.f__yx[self.pixelDistance < self.min_rad] = ma.masked
        if self.max_rad is not None:
            self.f__yx[self.pixelDistance > self.max_rad] = ma.masked
        else:
            self.max_rad = self.pixelDistance.max()


    @property
    def mask(self):
        return ~self.f__yx.mask
    
    
    def _getPixelDistance(self, x0, y0, pa, ba):
        pd = getImageDistance(self.f__yx.shape, x0, y0, pa, ba)
        return ma.array(pd, mask=self.f__yx.mask)


    def _assureFitted(self):    
        if self._model is None:
            raise Exception('Model not fitted yet.')


    def _getRadialProfileScatter(self):
        r = self.pixelDistance.compressed()
        sort_ix = np.argsort(r)
        r = r[sort_ix]
        f = self.f__yx.compressed()[sort_ix]
        var = self.var_f__yx.compressed()[sort_ix] if self.var_f__yx is not None else None
        return r, f, var
    

    def _getRadialProfileMean(self):
        bin_r = np.arange(self.min_rad, self.max_rad, self._pixel_fraction)
        bin_center = (bin_r[:-1] + bin_r[1:]) / 2.0
        if self.var_f__yx is None:
            f__r = radialProfile(self.f__yx, self.pixelDistance, bin_r, rad_scale=1.0, mode='mean')
            var__r = radialProfile(self.f__yx, self.pixelDistance, bin_r, rad_scale=1.0, mode='var')
        else:
            w__yx = 1.0 / self.var_f__yx
            var__r = 1.0 / radialProfile(w__yx, self.pixelDistance, bin_r, rad_scale=1.0, mode='sum')
            f__r = radialProfile(self.f__yx * w__yx, self.pixelDistance, bin_r, rad_scale=1.0, mode='sum') * var__r
            
        r = ma.array(bin_center, mask=f__r.mask)
        return r.compressed(), f__r.compressed(), var__r.compressed()
    

    def _getRadialProfile(self, mode='scatter'):
        if mode == 'scatter':
            return self._getRadialProfileScatter()
        elif mode == 'mean':
            return self._getRadialProfileMean()
        
        
    def fitModel(self, mode='scatter', guess=None):
        r, f, var = self._getRadialProfile(mode)
        if guess is not None:
            I_Be, R_e, I_D0, R_0 = guess[:4]
        else:
            I_Be=f.max()
            R_e=r.max()/4.0
            I_D0=f.max()/2.0
            R_0=r.max()/4.0

        if self.enable_bounds:
            bounds = {
                      'I_Be': [0.0, I_Be * self.tolerance],
                      'R_e': [0.0, R_e * self.tolerance],
                      'I_D0': [0.0, I_D0 * self.tolerance],
                      'R_0': [0.0, R_0 * self.tolerance],
                      }
            model = GalaxyModel(I_Be, R_e, I_D0, R_0, self._sigma, bounds=bounds)
            model.deriv = model.bound_deriv
        else:
            model = GalaxyModel(I_Be, R_e, I_D0, R_0, self._sigma)
            
        fit = fitting.NonLinearLSQFitter(model)
        fit(r, f, weights=1.0/var)
        self._model = model
        self.R2 = self._R2(r, f)


    def _R2(self, r, f):
        self._assureFitted()
        err = f - self._model(r)
        return 1.0 - (np.sum(err**2) / np.sum((f - f.mean())**2))


    def getFitParams(self):
        self._assureFitted()
        return tuple(self._model.parameters) + (self.R2, self.x0, self.y0, self.pa, self.ba)
    
    
    @property
    def I_Be(self):
        self._assureFitted()
        return self._model.I_Be[0]


    @property
    def R_e(self):
        self._assureFitted()
        return self._model.R_e[0]


    @property
    def I_D0(self):
        self._assureFitted()
        return self._model.I_D0[0]


    @property
    def R_0(self):
        self._assureFitted()
        return self._model.R_0[0]


    @property
    def sigma(self):
        self._assureFitted()
        return self._model.sigma[0]


    def plot_model(self, mode='scatter', figure=None, title='', interactive=True):
        if not __debug__:
            logger.warn('Plotting only works when running in debug mode.')
            return
        self._assureFitted()
        bulge_model = BulgeModel(self.I_Be, self.R_e)
        disk_model = DiskModel(self.I_D0, self.R_0)
        logger.debug(self._model)
        r, f, var = self._getRadialProfile(mode)
        import matplotlib.pyplot as plt
        plt.interactive(interactive)
        if figure is None:
            figure=plt.figure()        
        plt.clf()
        ax = plt.subplot(111)
        r_model = np.arange(r.min(), r.max(), 0.1)
        ax.errorbar(r, np.log10(f), yerr=np.sqrt(var)/f, fmt='o')
        ax.plot(r_model, np.log10(self._model(r_model)), 'r-')
        ax.plot(r_model, np.log10(PSF1d_gaussian_convolve(self._sigma, r_model, bulge_model)), 'r:')
        ax.plot(r_model, np.log10(PSF1d_gaussian_convolve(self._sigma, r_model, disk_model)), 'r--')
#         ax.set_ylim(np.log10(f.min()), np.log10(f.max()))
        ax.set_title(title)
        ax.text(0.75, 0.9, r'$R^2 = %.4f$' % self.R2, transform=ax.transAxes)
        ax.set_ylabel(r'$\log\ flux$')
        ax.set_xlabel(r'$r\ [arcsec]$')
        if not interactive:
            plt.show()
################################################################################
