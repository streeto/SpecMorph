'''
Created on 20/06/2014

@author: andre
'''

from pystarlight.util.velocity_fix import SpectraVelocityFixer
import numpy as np

__all__ = ['fix_kinematics']

def fix_kinematics(wl, flux, error, flags, v0, vd, target_vd, nproc=None):
    if vd is None:
        vd = np.zeros_like(v0)
        target_vd = 0    
    fixer = SpectraVelocityFixer(wl, v0, vd, nproc)
    flux, error, flags = fixer.fixFlagged(flux, error, flags, target_vd)
    return flux, error, flags
