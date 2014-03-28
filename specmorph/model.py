'''
Created on Jun 6, 2013

@author: andre
'''

from imfit import SimpleModelDescription, function_description

import numpy as np

__all__ = ['GalaxyModel']

################################################################################

def bulge_function(I_e, r_e, n, PA, ell):
    bulge = function_description('Sersic', name='bulge')
    bulge.I_e.setValue(I_e, [1e-33, 10*I_e])
    bulge.r_e.setValue(r_e, [1e-33, 10*r_e])
    bulge.n.setValue(n, [1,5])
    bulge.PA.setValue(PA, [-10, 190])
    bulge.ell.setValue(ell, [0, 1])
    return bulge

################################################################################

def disk_function(I_0, h, PA, ell):
    disk = function_description('Exponential', name='disk')
    disk.I_0.setValue(I_0, [1e-33, 10*I_0])
    disk.h.setValue(h, [1e-33, 10*h])
    disk.PA.setValue(PA, [-10, 190])
    disk.ell.setValue(ell, [0, 1])
    return disk

################################################################################

class GalaxyModel(SimpleModelDescription):

    def __init__(self, x0, y0, I_e, r_e, n, PA_b, ell_b, I_0, h, PA_d, ell_d):
        super(GalaxyModel, self).__init__()
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
        return GalaxyModel(x0=p['x0'], y0=p['y0'],
                           I_e=p['I_e'], r_e=p['r_e'], n=p['n'], PA_b=p['PA_b'], ell_b=p['ell_b'],
                           I_0=p['I_0'], h=p['h'], PA_d=p['PA_d'], ell_d=p['ell_d'])

        
    def getBulge(self):
        model = SimpleModelDescription()
        model.x0.value = self.x0.value
        model.y0.value = self.y0.value
        bulge = bulge_function(self.bulge.I_e.value, self.bulge.r_e.value, self.bulge.n.value,
                               self.bulge.PA.value, self.bulge.ell.value)
        model.addFunction(bulge)
        return model
        
        
    def getDisk(self):
        model = SimpleModelDescription()
        model.x0.value = self.x0.value
        model.y0.value = self.y0.value
        disk = disk_function(self.disk.I_0.value, self.disk.h.value,
                               self.disk.PA.value, self.disk.ell.value)
        model.addFunction(disk)
        return model
        
    
    @property
    def dtype(self):
        return np.dtype([('x0', 'float64'), ('y0', 'float64'),
                         ('I_e', 'float64'), ('r_e', 'float64'), ('n', 'float64'), ('PA_b', 'float64'), ('ell_b', 'float64'),
                         ('I_0', 'float64'), ('h', 'float64'), ('PA_d', 'float64'), ('ell_d', 'float64'),
                         ('flag', 'float64'), ('chi2', 'float64'), ('n_pix', 'float64'), ])


    def getParams(self):
        return (self.x0.value, self.y0.value,
                self.bulge.I_e.value, self.bulge.r_e.value, self.bulge.n.value, self.bulge.PA.value, self.bulge.ell.value,
                self.disk.I_0.value, self.disk.h.value, self.disk.PA.value, self.disk.ell.value,
                self.flag, self.chi2, self.nValidPixels)


    def __deepcopy__(self, memo):
        return type(self)(self.x0.value, self.y0.value,
                          self.bulge.I_e.value, self.bulge.r_e.value, self.bulge.n.value, self.bulge.PA.value, self.bulge.ell.value,
                          self.disk.I_0.value, self.disk.h.value, self.disk.PA.value, self.disk.ell.value)

################################################################################
