'''
Created on Jun 27, 2013

@author: andre
'''

from gauss_smooth import gaussVelocitySmooth  # @UnresolvedImport
from pycasso.util import logger
import numpy as np

__all__ = ['SpectraVelocityFixer']

################################################################################
class SpectraVelocityFixer(object):
    
    def __init__(self, l_obs, v_0, v_d, nproc=-1):
        self.l_obs = l_obs
        self.v_0 = np.asarray(v_0)
        self.v_d = np.asarray(v_d)
        self.nproc = nproc
        
        
    def _params(self, flux, v_d):
        if flux.ndim == 1:
            N_spec = 1
        else:
            N_spec = flux.shape[1]
        for i in xrange(N_spec):
            yield (self.l_obs, flux[:,i], self.v_0[i], v_d[i])
      
            
    def __call__(self, flux, target_vd=0.0):
        # Fix the velocity dispersion only if needed.
        m = self.v_d < target_vd
        vd_fix = np.zeros_like(self.v_d)
        vd_fix[m] = np.sqrt(target_vd**2 - self.v_d[m]**2)

        try:
            from joblib import Parallel, delayed
            f_fixed = Parallel(n_jobs=self.nproc)(delayed(fix_spectra)(args) for args in self._params(flux, vd_fix))
        except:
            logger.warn('joblib not installed, falling back to serial processing.')
            f_fixed = [fix_spectra(args) for args in self._params(flux, vd_fix)]

        return np.array(f_fixed).T


################################################################################
def fix_spectra(args):
    l_obs, flux, v_0, v_d = args
    return gaussVelocitySmooth(l_obs, flux, -v_0, v_d)
################################################################################


