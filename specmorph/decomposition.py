'''
Created on Jun 6, 2013

@author: andre
'''

from pycasso import fitsQ3DataCube
from pycasso.util import logger, getImageDistance
import numpy as np
from os import path, unlink
from astropy import nddata

from .fitting import BulgeDiskFitter
from .model import BulgeModel, DiskModel
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

    def __call__(self, fl__yx, var__yx=None):
        fitter = BulgeDiskFitter(fl__yx, var__yx)
        fitter.setup(x0=self.x0, y0=self.y0, sigma=self.sigma,
                     rad_clip_in=self.rad_clip_in, rad_clip_out=self.rad_clip_out)
        fitter.fitModel(self.mode)
        if self.plot:
            fitter.plot_model(self.mode, interactive=False)
        return fitter.getFitParams()
################################################################################


################################################################################
class ModelImageWrapper(object):
    
    def __init__(self, N_x, N_y, flux_unit, sigma, mask):
        self.shape = (N_y, N_x)
        self.flux_unit = flux_unit
        self.PSF = gaussian2d_kernel(sigma)
        self.mask = mask
    
    
    def _getModelImage(self, model, x0, y0, pa, ba):
        r__yx = getImageDistance(self.shape, x0, y0, pa, ba)
        image = np.ma.array(model(r__yx), mask=~self.mask)
        image.fill_value = np.nan
        return image


    def __call__(self, p):
        bulge_model = BulgeModel(p['I_Be'], p['R_e'])
        disk_model = DiskModel(p['I_D0'], p['R_0'])
        x0 = p['x0']
        y0 = p['y0']
        pa = p['pa']
        ba = p['ba']

        bulge = self._getModelImage(bulge_model, x0, y0, pa, ba).filled() * self.flux_unit
        # FIXME: bulge spectra does not behave at the center (r=0),
        # so we cheat by using the nddata.convolve "feature" of
        # interpolating around nan values.
        bulge[int(y0), int(x0)] = np.nan
        disk = self._getModelImage(disk_model, x0, y0, pa, ba).filled() * self.flux_unit

        bulge = nddata.convolve(bulge, self.PSF, boundary='extend')
        disk = nddata.convolve(disk, self.PSF, boundary='extend')
        return bulge, disk
################################################################################
        

################################################################################
def gaussian2d_kernel(sigma):
    center = int(4.0 * sigma)
    fw = 2 * center + 1
    xx, yy = np.indices((fw, fw), dtype='int')
    rr2 = (xx - center)**2 + (yy - center)**2
    return np.exp(- 0.5 * rr2 / sigma**2) / (2.0 * np.pi * sigma**2)
################################################################################



################################################################################
def picklehack(recarr):
    '''
    Recarray elements are not picklable, but a dict
    quacks close enough. See issue
    https://github.com/numpy/numpy/issues/3003
    '''
    for v in recarr:
        yield dict(zip(recarr.dtype.names, v))
################################################################################


################################################################################
class BulgeDiskDecomposition(fitsQ3DataCube):

    def __init__(self, synthesisFile, smooth=True, target_vd=0.0, FWHM=0.0, purge_cache=False, nproc=-1):
        self._nproc = nproc
        fitsQ3DataCube.__init__(self, synthesisFile, smooth)
        self._loadRestFrameSpectra(synthesisFile + '.rest-spectra.h5', target_vd, purge_cache)
        self.f_syn_rest__lyx = self.zoneToYX(self.f_syn_rest__lz, extensive=True, surface_density=False)
        self.f_obs_rest__lyx = self.zoneToYX(self.f_obs_rest__lz, extensive=True, surface_density=False)
        self.f_err_rest__lyx = self.zoneToYX(self.f_err_rest__lz, extensive=True, surface_density=False)
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
            f = self.f_syn_rest__lyx[l1] / self.flux_unit
            # Disabled error weighting
            # var = (self.f_err_rest__lyx[l1] / self.flux_unit)**2
            var = None
        else:
            # Disabled error weighting
            # w = self.flux_unit**2 / self.f_err_rest__lyx[l1:l2]**2
            # var = 1.0 / w.sum(axis=0)
            # f = (self.f_syn_rest__lyx[l1:l2] / self.flux_unit * w).sum(axis=0) * var
            f = np.mean(self.f_syn_rest__lyx[l1:l2], axis=0) / self.flux_unit
            var = None
        return f, var


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
            if plot: raise UserWarning('Plotting in parallel mode is no allowed, faking error.')
            from joblib import Parallel, delayed
            fit_params = Parallel(n_jobs=self._nproc)(delayed(morphology_fit)(fl, var) for fl, var in spec_slices)
        except (ImportError, UserWarning):
            logger.warn('Falling back to serial processing.')
            fit_params = [morphology_fit(fl, var) for fl, var in spec_slices]
        fit_params = np.array(fit_params, dtype=BulgeDiskFitter.getParamDtype())
        selected_wl_ix = np.arange(0, self.Nl_obs, step)
        return fit_params, selected_wl_ix
    
    
    def getModelSpectra(self, params, mask=None):
        if mask is None:
            mask = self.qMask
        model_image = ModelImageWrapper(self.N_x, self.N_y, self.flux_unit, self._sigma, mask)
        try:
            from joblib import Parallel, delayed
            result = Parallel(n_jobs=self._nproc)(delayed(model_image)(p) for p in picklehack(params))
        except (ImportError, UserWarning):
            logger.warn('Falling back to serial processing.')
            result = [model_image(p) for p in params]
            
        bulge, disk = zip(*result)
        return np.array(bulge), np.array(disk)
        
        
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

