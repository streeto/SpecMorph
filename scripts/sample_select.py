'''
Created on 10/03/2015

@author: andre
'''

from asciitable import CommentedHeader
import numpy as np
from glob import glob
from specmorph.califa.morph_class import load_morph_class
from pycasso.fitsdatacube import fitsQ3DataCube


#############################################################################
def get_califa_id(f):
    '''
    Return the CALIFA ID (as integer) from a PyCASSO datacube filename.
    '''
    from os import path
    base = path.basename(f)
    califa_id = base[1:5]
    return int(califa_id)
#############################################################################

#############################################################################
def get_ba(cubes):
    '''
    Return an array of b/a of the given cubes. 
    '''
    ba = []
    for f in cubes:
        print 'Computing b/a for %s' % f
        _K = fitsQ3DataCube(f)
        _pa, _ba = _K.getEllipseParams()
        ba.append(_ba)
    return np.array(ba)
#############################################################################

#############################################################################
def load_masterlist(fname):
    from califa import masterlist  # @UnusedImport
    import atpy
    
    t = atpy.Table(fname, type='califa_masterlist')
    
    # remove last item (Mice b)
    return t.rows(np.arange(len(t) - 1))
#############################################################################


#############################################################################
# Load the galaxy data.
#############################################################################

mc = load_morph_class('../data/tables/morph_eye_class.fits')

# Mark the available cubes as observed galaxies.
mc.add_column('observed', np.zeros(len(mc), dtype='int'))
obs_cubes = glob('../../cubes.px1/*_synthesis_eBR_px1_q043.d14a512.ps03.k1.mE.CCM.Bgsd6e.fits')
califa_id = np.array([get_califa_id(f) for f in obs_cubes])

# Make the CALIFA IDs into indices.
obs_keys = califa_id - 1
mc.observed[obs_keys] = 1

# Measure b/a.
mc.add_column('ba', np.ones(len(mc), dtype='bool') * -1.0)
mc.ba[obs_keys] = get_ba(obs_cubes)

ml = load_masterlist('../data/tables/califa_master_list_rgb.txt')
#############################################################################
# Select the sample.
#############################################################################

type_S0 = 8
ba_threshold = 0.7

#sample = t.observed == 1
#sample &= t.ba >= ba_threshold
sample = ml.ba >= ba_threshold
sample &= mc.type_max >= type_S0
sample &= mc.type_min <= type_S0
sample &= mc.merger == 0
sample &= mc.barred == 0

t_sample = mc.where(sample)
t_sample_fname = '../data/tables/sample.txt'
print 'Writing sample table %s' % t_sample_fname
t_sample.write(t_sample_fname, type='ascii', Writer=CommentedHeader, overwrite=True)

