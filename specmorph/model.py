'''
Created on Jun 6, 2013

@author: andre
'''

from imfit import SimpleModelDescription, function_description, parse_config_file
from pycasso.util import logger
from .geometry import ellipse_params, distance, r50
from .fitting import fit_image

import numpy as np

__all__ = ['BDModel', 'bd_initial_model']

################################################################################

def bd_initial_model(image, noise, PSF, x0=None, y0=None):
    '''
    Doc me!
    '''
    if x0 is None: x0 = image.shape[1] / 2.0
    if y0 is None: y0 = image.shape[0] / 2.0
    pa, ba = ellipse_params(image, x0, y0)
    ell = 1.0 - ba
    pa = pa * 180.0 / np.pi
    if pa < 0.0:
        pa += 180.0
    r = distance(image.shape, x0, y0, pa, ba)
    r = np.ma.array(r, mask=image.mask)
    image_r50 = r50(image, r)
    disk_begins = 1.5 * image_r50
    disk_image = image.copy()
    disk_image[r < disk_begins] = np.ma.masked
    disk_noise = noise.copy()
    disk_noise[r < disk_begins] = np.ma.masked
    guess_I_0 = disk_image.max() * 2.0
    disk_model = BDModel(wl=5635.0, x0=x0, y0=y0,
                         I_e=0.0, r_e=image_r50/2.0, n=3, PA_b=pa, ell_b=ell,
                         I_0=disk_image.max(), h=disk_begins, PA_d=pa, ell_d=ell)
    disk_model.disk.I_0.limits = (1e-33, 5* guess_I_0)
    disk_model.bulge.I_e.fixed=True
    disk_model.bulge.r_e.fixed=True
    disk_model.bulge.n.fixed=True
    disk_model.bulge.PA.fixed=True
    disk_model.bulge.ell.fixed=True
    logger.debug('Initial guess for disk (r > %.2f):\n%s\n' % (disk_begins, str(disk_model)))
    fitted_model, converged, chi2 = fit_image(disk_image, disk_noise, disk_model, PSF, mode='NM')
    logger.info('Disk fit - converged: %s; chi2 = %.2f; h = %.2f' % (converged, chi2, fitted_model.disk.h.value))
    logger.debug('Fitted disk (r > %.2f):\n%s\n' % (disk_begins, str(fitted_model)))
    bdmodel = BDModel(wl=5635.0, x0=x0, y0=y0,
                      I_e=image.max(), r_e=image_r50/2.0, n=3, PA_b=pa, ell_b=ell,
                      I_0=fitted_model.disk.I_0.value, h=fitted_model.disk.h.value,
                      PA_d=fitted_model.disk.PA.value, ell_d=fitted_model.disk.ell.value)
    bdmodel.disk.I_0.setTolerance(0.3)
    bdmodel.disk.h.setTolerance(0.3)
    logger.debug('Guess model:\n%s\n' % str(bdmodel))
    initial_model, converged, chi2 = fit_image(image, noise, bdmodel, PSF, mode='DE', quiet=False)
    logger.info('Initial model fit - converged: %s; chi2 = %.2f' % (converged, chi2))
    logger.debug('Initial model:\n%s\n' % str(initial_model))
    return initial_model

################################################################################

def bulge_function(I_e, r_e, n, PA, ell):
    bulge = function_description('Sersic', name='bulge')
    bulge.I_e.setValue(I_e, [1e-33, 2.0*I_e])
    bulge.r_e.setValue(r_e, [1e-33, 2.0*r_e])
    bulge.n.setValue(n, [1.0,5.0])
    bulge.PA.setValue(PA, [-1.0, 181.0])
    bulge.ell.setValue(ell, [0.0, 1.0])
    return bulge

################################################################################

def disk_function(I_0, h, PA, ell):
    disk = function_description('Exponential', name='disk')
    disk.I_0.setValue(I_0, [1e-33, 2.0*I_0])
    disk.h.setValue(h, [1e-33, 2.0*h])
    disk.PA.setValue(PA, [-1.0, 181.0])
    disk.ell.setValue(ell, [0.0, 1.0])
    return disk

################################################################################

class BDModel(SimpleModelDescription):

    def __init__(self, wl, x0, y0, I_e, r_e, n, PA_b, ell_b, I_0, h, PA_d, ell_d):
        super(BDModel, self).__init__()
        self.wl = wl
        self.x0.setValue(x0, [x0-10, x0+10])
        self.y0.setValue(y0, [y0-10, y0+10])
        self.flag = 0.0
        self.chi2 = 0.0
        self.nValidPixels = 0
        if I_e is not None and r_e is not None:
            bulge = bulge_function(I_e, r_e, n, PA_b, ell_b)
            self.addFunction(bulge)
        if I_0 is not None and h is not None:
            disk = disk_function(I_0, h, PA_d, ell_d)
            self.addFunction(disk)


    @classmethod
    def fromParamVector(cls, p):
        return BDModel(wl=p['wl'], x0=p['x0'], y0=p['y0'],
                       I_e=p['I_e'], r_e=p['r_e'], n=p['n'], PA_b=p['PA_b'], ell_b=p['ell_b'],
                       I_0=p['I_0'], h=p['h'], PA_d=p['PA_d'], ell_d=p['ell_d'])

        
    @classmethod
    def readConfig(cls, config):
        model = parse_config_file(config)
        wl = 0.0
        x0 = model.fs0.x0.value
        y0 = model.fs0.y0.value
        I_e = model.fs0.Sersic.I_e.value
        r_e = model.fs0.Sersic.r_e.value
        n = model.fs0.Sersic.n.value
        PA_b = model.fs0.Sersic.PA.value
        ell_b = model.fs0.Sersic.ell.value
        I_0 = model.fs0.Exponential.I_0.value
        h = model.fs0.Exponential.h.value
        PA_d = model.fs0.Exponential.PA.value
        ell_d = model.fs0.Exponential.ell.value
        gmodel = cls(wl, x0, y0, I_e, r_e, n, PA_b, ell_b,
                     I_0, h, PA_d, ell_d)
        return gmodel


    def getBulge(self):
        model = SimpleModelDescription()
        model.wl = self.wl
        model.x0.value = self.x0.value
        model.y0.value = self.y0.value
        bulge = bulge_function(self.bulge.I_e.value, self.bulge.r_e.value, self.bulge.n.value,
                               self.bulge.PA.value, self.bulge.ell.value)
        model.addFunction(bulge)
        return model
        
        
    def getDisk(self):
        model = SimpleModelDescription()
        model.wl = self.wl
        model.x0.value = self.x0.value
        model.y0.value = self.y0.value
        disk = disk_function(self.disk.I_0.value, self.disk.h.value,
                               self.disk.PA.value, self.disk.ell.value)
        model.addFunction(disk)
        return model
        
    
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
        return type(self)(self.wl, self.x0.value, self.y0.value,
                          self.bulge.I_e.value, self.bulge.r_e.value, self.bulge.n.value, self.bulge.PA.value, self.bulge.ell.value,
                          self.disk.I_0.value, self.disk.h.value, self.disk.PA.value, self.disk.ell.value)

################################################################################
