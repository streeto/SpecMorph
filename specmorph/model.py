'''
Created on Jun 6, 2013

@author: andre
'''

from .geometry import ellipse_params, distance, r50, fix_PA_ell
from .fitting import fit_image, model_image
from .util import logger

from imfit import SimpleModelDescription, function_description, parse_config_file
import numpy as np
from os import unlink, path

__all__ = ['BDModel', 'bulge_model', 'disk_model', 'bd_initial_model', 'create_model_images']

################################################################################
def bd_initial_model(image, noise, PSF, x0=None, y0=None, quiet=True, nproc=0,
                     use_cash_statitics=False, cache_model_file=None):
    '''
    Doc me!
    '''
    if cache_model_file is not None and path.exists(cache_model_file):
        try:
            initial_model = BDModel.load(cache_model_file)
            logger.debug('Cached model found:\n%s\n' % initial_model)
            return initial_model
        except:
            logger.warn('Bad cache model file %s. Deleting.' % cache_model_file)
            unlink(cache_model_file)
    initial_model = _bd_initial_model(image, noise, PSF, quiet=quiet)
    if cache_model_file is not None:
        with open(cache_model_file, 'w') as f:
            logger.debug('Saving cache model %s.' % cache_model_file)
            try:
                f.write(str(initial_model))
            except:
                logger.warn('Could not write cache model file %s' % cache_model_file)
    return initial_model
################################################################################


################################################################################
def _bd_initial_model(image, noise, PSF, x0=None, y0=None, quiet=True, nproc=0, use_cash_statitics=False):
    '''
    Doc me!
    '''
    if x0 is None: x0 = image.shape[1] / 2.0
    if y0 is None: y0 = image.shape[0] / 2.0
    pa, ell = ellipse_params(image, x0, y0)
    r = distance(image.shape, x0, y0, pa, ell)
    r = np.ma.array(r, mask=image.mask)
    image_r50 = r50(image, r)
    disk_begins = 1.0 * image_r50
    disk_image = image.copy()
    disk_image[r < disk_begins] = np.ma.masked
    disk_noise = noise.copy()
    disk_noise[r < disk_begins] = np.ma.masked
    guess_I_0 = disk_image.max() * 1.0
    dmodel = disk_model(wl=5635.0, x0=x0, y0=y0, I_0=guess_I_0, h=disk_begins, PA=pa, ell=ell)
    dmodel.x0.setLimitsRel(5, 5)
    dmodel.y0.setLimitsRel(5, 5)
    dmodel.disk.I_0.setLimits(1e-33, 5.0 * guess_I_0)
    dmodel.disk.h.setLimits(1e-33, 2.5 * disk_begins)
    dmodel.disk.PA.setLimitsRel(30.0, 30.0)
    dmodel.disk.ell.setLimitsRel(0.25, 0.25)
    logger.debug('Initial guess for disk (r > %.2f):\n%s\n' % (disk_begins, str(dmodel)))
    f_dmodel, converged, chi2 = fit_image(disk_image, disk_noise, dmodel, PSF,
                                                   mode='DE', quiet=quiet, nproc=nproc,
                                                   use_cash_statistics=use_cash_statitics)
    logger.info('Disk fit - converged: %s; chi2 = %.2f; h = %.2f' % (converged, chi2, f_dmodel.disk.h.value))
    logger.debug('Fitted disk (r > %.2f):\n%s\n' % (disk_begins, str(f_dmodel)))

    image_max = image.max()
    pa, ell = fix_PA_ell(f_dmodel.disk.PA.value, f_dmodel.disk.ell.value)
    bdmodel = BDModel()
    bdmodel.wl = 5635.0
    bdmodel.x0.setValue(x0)
    bdmodel.x0.setLimitsRel(5, 5)
    bdmodel.y0.setValue(y0)
    bdmodel.y0.setLimitsRel(5, 5)
    bdmodel.disk.I_0.setValue(f_dmodel.disk.I_0.value)
    bdmodel.disk.I_0.setTolerance(0.5)
    bdmodel.disk.h.setValue(f_dmodel.disk.h.value)
    bdmodel.disk.h.setTolerance(0.5)
    bdmodel.disk.PA.setValue(pa)
    bdmodel.disk.PA.setLimitsRel(10.0, 10.0)
    bdmodel.disk.ell.setValue(ell)
    bdmodel.disk.ell.setLimitsRel(0.1, 0.1)

    bdmodel.bulge.I_e.setValue(image_max)
    bdmodel.bulge.I_e.setLimits(1e-33, 3.0 * image_max)
    bdmodel.bulge.r_e.setValue(image_r50 / 2.0)
    bdmodel.bulge.r_e.setLimits(1e-33, 2.5 * image_r50)
    bdmodel.bulge.n.setValue(3.0, vmin=1.0, vmax=5.0)
    bdmodel.bulge.PA.setValue(pa)
    bdmodel.bulge.PA.setLimitsRel(30.0, 30.0)
    bdmodel.bulge.ell.setValue(ell)
    bdmodel.bulge.ell.setLimitsRel(0.25, 0.25)
    
    logger.debug('First guess model:\n%s\n' % str(bdmodel))
    bdmodel, converged, chi2 = fit_image(image, noise, bdmodel, PSF,
                                         mode='DE', quiet=quiet, nproc=nproc,
                                         use_cash_statistics=use_cash_statitics)
    logger.info('First guess fit - converged: %s; chi2 = %.2f' % (converged, chi2))

    pa, ell = fix_PA_ell(bdmodel.bulge.PA.value, bdmodel.bulge.ell.value)
    bdmodel.bulge.PA.setValue(pa)
    bdmodel.bulge.PA.setLimitsRel(10.0, 10.0)
    bdmodel.bulge.ell.setValue(ell)
    bdmodel.bulge.ell.setLimitsRel(0.1, 0.1)
    pa, ell = fix_PA_ell(bdmodel.disk.PA.value, bdmodel.disk.ell.value)
    bdmodel.disk.PA.setValue(pa)
    bdmodel.disk.PA.setLimitsRel(10.0, 10.0)
    bdmodel.disk.ell.setValue(ell)
    bdmodel.disk.ell.setLimitsRel(0.1, 0.1)
    logger.debug('Second guess model:\n%s\n' % str(bdmodel))
    bdmodel, converged, chi2 = fit_image(image, noise, bdmodel, PSF,
                                         mode='NM', quiet=quiet, nproc=nproc,
                                         use_cash_statistics=use_cash_statitics)
    logger.info('Second guess - converged: %s; chi2 = %.2f' % (converged, chi2))

    bdmodel.disk.I_0.setLimits(1e-33, 10.0 * bdmodel.disk.I_0.value)
    bdmodel.disk.h.setTolerance(0.3)
    bdmodel.disk.PA.setLimitsRel(10.0, 10.0)
    bdmodel.disk.ell.setLimitsRel(0.1, 0.1)
    bdmodel.disk.I_0.setLimits(1e-33, 10.0 * bdmodel.bulge.I_e.value)
    bdmodel.bulge.r_e.setTolerance(0.3)
    bdmodel.bulge.n.setTolerance(0.5)
    bdmodel.bulge.PA.setLimitsRel(10.0, 10.0)
    bdmodel.bulge.ell.setLimitsRel(0.1, 0.1)
    
    logger.debug('Final model:\n%s\n' % str(bdmodel))
    return bdmodel
################################################################################


################################################################################
def create_model_images(model, shape, PSF, nproc=None):
    bulge_model = model.getBulge()
    disk_model = model.getDisk()
    
    bulge = model_image(bulge_model, shape, PSF, nproc)
    disk = model_image(disk_model, shape, PSF, nproc)
    return bulge, disk
################################################################################

    
################################################################################
def bulge_function(I_e=0.0, r_e=0.0, n=0.0, PA=0.0, ell=0.0):
    bulge = function_description('Sersic', name='bulge')
    bulge.I_e.setValue(I_e)
    bulge.r_e.setValue(r_e)
    bulge.n.setValue(n)
    bulge.PA.setValue(PA)
    bulge.ell.setValue(ell)
    return bulge
################################################################################


################################################################################
def disk_function(I_0=0.0, h=0.0, PA=0.0, ell=0.0):
    disk = function_description('Exponential', name='disk')
    disk.I_0.setValue(I_0)
    disk.h.setValue(h)
    disk.PA.setValue(PA)
    disk.ell.setValue(ell)
    return disk
################################################################################


################################################################################
def smooth_param_polynomial(param, wl, flags, l_obs, degree=1):
    flag_ok = (flags == 0) & (wl > 4500.0)
    from astropy.modeling import models, fitting
    if degree == 0:
        line = models.Polynomial1D(degree=1)
        line.c1 = 0.0
        line.c1.fixed = True
    else:
        line = models.Polynomial1D(degree)
    fit = fitting.LinearLSQFitter()
    param_fitted = fit(line, wl[flag_ok], param[flag_ok])
    return param_fitted(l_obs)
################################################################################


################################################################################
def smooth_models(models, wl, degree=1, fix_structural=True):
    params = np.array([m.getParams() for m in models], dtype=models[0].dtype)
    smooth_params = np.empty(len(wl), dtype=params[0].dtype)    
    param_wl = params['wl']
    param_flag = params['flag']

    for p in params.dtype.names:
        if p in ['wl', 'flag', 'chi2', 'n_pix']: continue
        smooth_params[p] = smooth_param_polynomial(params[p], param_wl, param_flag, wl, degree)
    
    models = []
    for i in xrange(len(smooth_params)):
        m = BDModel.fromParamVector(smooth_params[i])
        if fix_structural:
            m.x0.fixed=True
            m.y0.fixed=True
            m.bulge.r_e.fixed=True
            m.bulge.n.fixed=True
            m.bulge.PA.fixed=True
            m.bulge.ell.fixed=True
            m.disk.h.fixed=True
            m.disk.PA.fixed=True
            m.disk.ell.fixed=True
        models.append(m)
    return models
################################################################################


################################################################################
def disk_model(wl, x0, y0, I_0, h, PA, ell):
    model = SimpleModelDescription()
    model.wl = wl
    model.x0.setValue(x0)
    model.y0.setValue(y0)
    disk = disk_function(I_0, h, PA, ell)
    model.addFunction(disk)
    return model
################################################################################


################################################################################
def bulge_model(wl, x0, y0, I_e, r_e, n, PA, ell):
    model = SimpleModelDescription()
    model.wl = wl
    model.x0.setValue(x0)
    model.y0.setValue(y0)
    bulge = bulge_function(I_e, r_e, n, PA, ell)
    model.addFunction(bulge)
    return model
################################################################################


################################################################################
class BDModel(SimpleModelDescription):

    def __init__(self, inst=None):
        super(BDModel, self).__init__(inst)
        self.wl = 0.0
        self.flag = 0.0
        self.chi2 = 0.0
        self.nValidPixels = 0
        if inst is not None:
            # Rename loaded functions.
            self.Sersic._name = 'bulge'
            self.Exponential._name = 'disk'
        else:
            bulge = bulge_function()
            self.addFunction(bulge)
            disk = disk_function()
            self.addFunction(disk)


    @classmethod
    def fromParamVector(cls, p):
        model = cls()
        model.wl = p['wl']
        model.x0.setValue(p['x0'])
        model.y0.setValue(p['y0'])
        model.bulge.I_e.setValue(p['I_e'])
        model.bulge.r_e.setValue(p['r_e'])
        model.bulge.n.setValue(p['n'])
        model.bulge.PA.setValue(p['PA_b'])
        model.bulge.ell.setValue(p['ell_b'])
        model.disk.I_0.setValue(p['I_0'])
        model.disk.h.setValue(p['h'])
        model.disk.PA.setValue(p['PA_d'])
        model.disk.ell.setValue(p['ell_d'])
        return model

        
    @classmethod
    def load(cls, config):
        model = SimpleModelDescription.load(config)
        model = cls(model)
        return model


    def getBulge(self):
        return bulge_model(self.wl, self.x0.value, self.y0.value,
                           self.bulge.I_e.value, self.bulge.r_e.value, self.bulge.n.value,
                           self.bulge.PA.value, self.bulge.ell.value)
        
        
    def getDisk(self):
        return disk_model(self.wl, self.x0.value, self.y0.value,
                          self.disk.I_0.value, self.disk.h.value,
                          self.disk.PA.value, self.disk.ell.value)
        
    
    @property
    def dtype(self):
        return np.dtype([('wl', 'float64'), ('x0', 'float64'), ('y0', 'float64'),
                         ('I_e', 'float64'), ('r_e', 'float64'), ('n', 'float64'), ('PA_b', 'float64'), ('ell_b', 'float64'),
                         ('I_0', 'float64'), ('h', 'float64'), ('PA_d', 'float64'), ('ell_d', 'float64'),
                         ('flag', 'float64'), ('chi2', 'float64'), ('n_pix', 'float64'), ])


    def getParams(self):
        return (self.wl, self.x0.value, self.y0.value,
                self.bulge.I_e.value, self.bulge.r_e.value, self.bulge.n.value, self.bulge.PA.value, self.bulge.ell.value,
                self.disk.I_0.value, self.disk.h.value, self.disk.PA.value, self.disk.ell.value,
                self.flag, self.chi2, self.nValidPixels)


    def __deepcopy__(self, memo):
        model = super(BDModel, self).__deepcopy__(memo)
        model.wl = self.wl
        model.flag = self.flag
        model.chi2 = self.chi2
        model.nValidPixels = self.nValidPixels
        return model

################################################################################
