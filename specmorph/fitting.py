'''
Created on Jun 10, 2013

@author: andre
'''

from astropy.modeling import fitting
from pycasso.util import getEllipseParams, getImageDistance, radialProfile, logger
import numpy as np
from numpy import ma
from .model import BulgeModel, DiskModel, GalaxyModel

__all__ = ['BulgeDiskFitter', 'FitterInput', 'flag_ok', 'flag_bad', 'flag_out_of_bounds', 'flag_max_iter_reached']

enable_plot=False

################################################################################
flag_ok = 0
flag_bad = 1
flag_out_of_bounds = 2
flag_max_iter_reached = 4
################################################################################


################################################################################
class FitterInput(object):
    
    def __init__(self, wl, f__yx, var_f__yx=None, x0=None, y0=None, ba=None, pa=None):
        self.wl = wl
        self.f__yx = f__yx
        self.var_f__yx = var_f__yx
        self.x0 = x0
        self.y0 = y0
        self.ba = ba
        self.pa = pa
################################################################################


################################################################################
class BulgeDiskFitter(object):
    
    _pixel_fraction = 0.02
    _maxiter = 400
    almost_zero = 1e-20
    _param_dtype = np.dtype([('I_Be', 'float64'), ('R_e', 'float64'),
                            ('I_D0', 'float64'), ('R_0', 'float64'),
                            ('R2', 'float64'),
                            ('x0', 'float64'), ('y0', 'float64'),
                            ('pa', 'float64'), ('ba', 'float64'),
                            ('flag', 'int32')])
        

    @classmethod
    def getNparams(cls):
        return len(cls._param_dtype.fields)
        
        
    @classmethod
    def getParamDtype(cls):
        return cls._param_dtype
        
        
    def __init__(self, sigma=0.0, min_rad=0.0, radprof_mode='scatter', enable_bounds=True,
                 use_deriv=True, fitter='leastsq'):
        self._radprof_mode = radprof_mode
        self._enable_bounds = enable_bounds
        self._use_deriv = use_deriv
        if fitter == 'leastsq':
            logger.debug('Fitting using Levenberg-Marquardt algorithm (scipy.optimize.leastsq).')
            self._fitter = fitting.NonLinearLSQFitter
        elif fitter == 'slsqp':
            logger.debug('Fitting using Sequential Least Squares Programming (scipy.optimize.slsqp).')
            self._fitter = fitting.SLSQPFitter
        else:
            raise ValueError('Bad fitter: %s' % fitter)

        self._sigma = sigma
        self._min_rad = min_rad
        self._max_rad = None
        self.pixelDistance = None
        self._radial_scale = None

        self.wl = None
        self.f__yx = None
        self.var_f__yx = None
        self._flux_scale = None

        self.x0 = 0.0
        self.y0 = 0.0
        self.pa = 0.0
        self.ba = 1.0

        self.R2 = np.nan
        self.flag = flag_ok
        self._model = None


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
        r = self.pixelDistance.compressed() / self._radial_scale
        sort_ix = np.argsort(r)
        r = r[sort_ix]
        f = self.f__yx.compressed()[sort_ix]
        var = self.var_f__yx.compressed()[sort_ix] if self.var_f__yx is not None else None
        return r, f, var
    

    def _getRadialProfileMean(self):
        bin_r = np.arange(self._min_rad, self._max_rad, self._pixel_fraction)
        bin_center = (bin_r[:-1] + bin_r[1:]) / 2.0
        if self.var_f__yx is None:
            f__r = radialProfile(self.f__yx, self.pixelDistance, bin_r, rad_scale=self._radial_scale, mode='mean')
            var__r = radialProfile(self.f__yx, self.pixelDistance, bin_r, rad_scale=self._radial_scale, mode='var')
            # A value of zero for variance will cause a infinite weight
            # in the fitting later, let's avoid that.
            invalid = (var__r == 0.0) & ~(var__r.mask)
            var__r[invalid] = np.min(var__r[~invalid])
        else:
            w__yx = 1.0 / self.var_f__yx
            var__r = 1.0 / radialProfile(w__yx, self.pixelDistance, bin_r, rad_scale=self._radial_scale, mode='sum')
            f__r = radialProfile(self.f__yx * w__yx, self.pixelDistance, bin_r, rad_scale=self._radial_scale, mode='sum') * var__r
            
        r = ma.array(bin_center, mask=f__r.mask)
        return r.compressed(), f__r.compressed(), var__r.compressed()
    

    def _getRadialProfile(self, mode='scatter'):
        if mode == 'scatter':
            return self._getRadialProfileScatter()
        elif mode == 'mean':
            return self._getRadialProfileMean()
        
        
    def _setup(self, fi):
        self.wl = fi.wl
        self._flux_scale = fi.f__yx.max()
        self.f__yx = fi.f__yx / self._flux_scale
        self.var_f__yx = fi.var_f__yx / self._flux_scale**2 if fi.var_f__yx is not None else None

        self.x0 = fi.x0
        self.y0 = fi.y0
        if self.x0 is None or self.y0 is None:
            # FIXME: perform a proper centering
            self.y0, self.x0 = np.where(self.f__yx == self.f__yx.max())[0]

        self.pa = fi.pa
        self.ba = fi.ba
        if self.pa is None or self.ba is None:
            self.pa, self.ba = getEllipseParams(self.f__yx, self.x0, self.y0, mask=self.mask)
        self.pixelDistance = self._getPixelDistance(self.x0, self.y0, self.pa, self.ba)

        # FIXME: use circular aperture for radial clip.
        self.f__yx[self.pixelDistance < self._min_rad] = ma.masked
        self._max_rad = self.pixelDistance.max()
        self._radial_scale = self._max_rad


    def __call__(self, fitter_input):
        self._setup(fitter_input)
        logger.debug('Fitting for wavelength: %.0f \\AA' % self.wl)
        r, f, var = self._getRadialProfile(self._radprof_mode)
        assert(r.shape == f.shape)
        assert(var is None or var.shape == f.shape)
        
        # Initial guess
        I_Be = 1.0 / (2.0 * np.exp(-7.669))
        R_e = 1.0 / 4.0
        I_D0 = 1.0 / 2.0
        R_0 = 1.0 / 4.0

        if self._enable_bounds:
            logger.debug('Bounds are enabled.')
            bounds = {
                      'I_Be': [self.almost_zero, None],
                      'R_e': [self.almost_zero, None],
                      'I_D0': [self.almost_zero, None],
                      'R_0': [self.almost_zero, None],
                      }
        else:
            bounds = None

        self._model = GalaxyModel(I_Be, R_e, I_D0, R_0, self._sigma / self._radial_scale, bounds=bounds)
        if not self._use_deriv:
            logger.debug('Disabled derivative function.')
            self._model.deriv = None
            
        fit = self._fitter(self._model)
        weights = 1.0/var if var is not None else None
        fit(r, f, weights=weights, maxiter=self._maxiter)
        self.R2 = self._R2(r, f)
        self._checkFit(fit)
        return self.getFitParams()


    def _insideBounds(self):
        param_array = np.array([self.I_Be, self.R_e, self.I_D0, self.R_0])
        return (param_array > self.almost_zero).all()

        
    def _checkFit(self, fit):
        self.flag = flag_ok
        fit_info = fit.fit_info
        if isinstance(fit, fitting.NonLinearLSQFitter):
            fit_error = fit_info['ierr'] not in [1, 2, 3, 4]
            n_iter = fit_info['nfev']
            message = fit_info['message']
        elif isinstance(fit, fitting.SLSQPFitter):
            fit_error = fit_info['exit_mode'] != 0
            n_iter = fit_info['numiter']
            message = fit_info['message']
        else:
            raise NotImplementedError('Unknown fitter type: %s' % type(fit))
        if fit_error:
            logger.warn('Error fitting model, message: %s' % message)
            self.flag |= flag_bad

        if self._enable_bounds and self._insideBounds():
            logger.warn('The fit stepped out of bounds:')
            logger.warn(self._model.parameters)
            logger.warn(fit.bounds)
            self.flag |= flag_out_of_bounds
            
        if n_iter >= self._maxiter:
            logger.warn('Maximum iterations reached: %d' % n_iter)
            self.flag |= flag_max_iter_reached

        if __debug__ and enable_plot:
            logger.warn('Plotting model')
            plot_title = 'Radprof. mode: %s | $\lambda = %.0f$' % (self._radprof_mode, self.wl)
            self.plot_model(self._radprof_mode, title=plot_title, interactive=False)


    def _R2(self, r, f):
        self._assureFitted()
        err = f - self._model(r)
        return 1.0 - (np.sum(err**2) / np.sum((f - f.mean())**2))


    def getFitParams(self):
        self._assureFitted()
        return (self.I_Be, self.R_e, self.I_D0, self.R_0,
                self.R2, self.x0, self.y0, self.pa, self.ba, self.flag)
    
    
    @property
    def I_Be(self):
        self._assureFitted()
        return self._model.I_Be.value * self._flux_scale


    @property
    def R_e(self):
        self._assureFitted()
        return self._model.R_e.value * self._radial_scale


    @property
    def I_D0(self):
        self._assureFitted()
        return self._model.I_D0.value * self._flux_scale


    @property
    def R_0(self):
        self._assureFitted()
        return self._model.R_0.value * self._radial_scale


    def plot_model(self, radprof_mode='scatter', figure=None, title='', interactive=True):
        if not __debug__:
            logger.warn('Plotting only works when running in debug mode.')
            return
        self._assureFitted()
        
        bulge_model = BulgeModel(self.I_Be, self.R_e, self._sigma)
        disk_model = DiskModel(self.I_D0, self.R_0, self._sigma)
        galaxy_model = GalaxyModel(self.I_Be, self.R_e, self.I_D0, self.R_0, self._sigma)
        logger.debug(self._model)
        r, f, var = self._getRadialProfile(radprof_mode)
        r *= self._radial_scale
        f *= self._flux_scale
        logger.debug('Radial scale = %.2f' % self._radial_scale)
        var = var * self._flux_scale**2 if var is not None else None
        
        import matplotlib.pyplot as plt
        plt.interactive(interactive)
        if figure is None:
            figure=plt.figure()        
        plt.clf()
        ax = plt.subplot(111)

        # Test for comparison between radial profile of model and data.
#         galaxy_model_test = GalaxyModel(self.I_Be, self.R_e, self.I_D0, self.R_0, 0.0)
#         model_image = galaxy_model_test(self.pixelDistance)
#         model_image[self.y0, self.x0] = np.nan
#         PSF = gaussian2d_kernel(self._sigma)
#         model_image = nddata.convolve(model_image, PSF, boundary='extend')
#         model_image = np.ma.array(model_image, mask=~self.mask, fill_value=np.nan)
#         r_test = self.pixelDistance.compressed()
#         sort_ix = np.argsort(r_test)
#         r_test = r_test[sort_ix]
#         f_test = model_image.compressed()[sort_ix]
        
        r_model = np.linspace(r.min(), r.max(), 100)
        if var is not None:
            ax.errorbar(r, np.log10(f), yerr=np.sqrt(var)/f, fmt='bo')
        else:
            ax.plot(r, np.log10(f), 'bo')
#             ax.plot(r_test, np.log10(f_test), 'gx')
        ax.plot(r_model, np.log10(galaxy_model(r_model)), 'r-')
        ax.plot(r_model, np.log10(bulge_model(r_model)), 'r:')
        ax.plot(r_model, np.log10(disk_model(r_model)), 'r--')
        ax.set_title(title)
        ax.text(0.75, 0.9, r'$R^2 = %.4f$' % self.R2, transform=ax.transAxes)
        ax.set_ylabel(r'$\log\ flux$')
        ax.set_xlabel(r'$r\ [arcsec]$')
        if not interactive:
            plt.show()
################################################################################
