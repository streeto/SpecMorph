'''
Created on 10/03/2015

@author: andre
'''

from asciitable import CommentedHeader
import numpy as np
from glob import glob
from specmorph.califa.tables import load_morph_class, load_masterlist
from specmorph.califa.tables import califa_id_from_cube, califa_id_to_int
from pycasso.fitsdatacube import fitsQ3DataCube
from os import path
import argparse


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
def parse_args():
    parser = argparse.ArgumentParser(description='Spectral Morphological Decomposition - sample select.')
    
    parser.add_argument('--sample-id', dest='sampleId', default='sample',
                        help='Sample name, will become the table file name.')
    parser.add_argument('--tables-dir', dest='tablesDir', default='data/tables',
                        help='Tables directory.')
    parser.add_argument('--cubes-dir', dest='cubesDir', default='../cubes.px1/',
                        help='PyCASSO cubes directory.')
    parser.add_argument('--ba-threshold', dest='baThreshold', type=float, default=0.7,
                    help='Minimum b/a in the sample.')
    parser.add_argument('--measure-ba', dest='measureBa', action='store_true',
                    help='Measure b/a instead of using the Master List.')
    args = parser.parse_args()
    return args

#############################################################################
# Load the galaxy data.
#############################################################################

args = parse_args()

mc = load_morph_class(path.join(args.tablesDir, 'morph_eye_class.fits'))

# Mark the available cubes as observed galaxies.
obs_cubes = glob(path.join(args.cubesDir, '*_synthesis_eBR_px1_q043.d14a512.ps03.k1.mE.CCM.Bgsd6e.fits'))
obs_califa_str = np.array([califa_id_from_cube(f) for f in obs_cubes])
obs_califa_id = np.array([califa_id_to_int(c_id) for c_id in obs_califa_str])
# Make the CALIFA IDs into indices.
obs_keys = obs_califa_id - 1

mc.add_column('observed', np.zeros(len(mc), dtype='int'))
mc.observed[obs_keys] = 1

mc.add_column('cube', np.zeros(len(mc), dtype='S128'))
mc.cube[obs_keys] = obs_cubes

if args.measureBa:
    mc.add_column('ba', np.ones(len(mc), dtype='bool') * -1.0)
    mc.ba[obs_keys] = get_ba(obs_cubes)

ml = load_masterlist(path.join(args.tablesDir, 'califa_master_list_rgb.txt'))
#############################################################################
# Select the sample.
#############################################################################

type_S0 = 8

sample = mc.observed == 1

if args.measureBa:
    sample &= mc.ba >= args.baThreshold
else:
    sample &= ml.ba >= args.baThreshold
    
sample &= mc.type_max >= type_S0
sample &= mc.type_min <= type_S0
sample &= mc.merger == 0
sample &= mc.barred == 0

t_sample = mc.where(sample)
t_sample_fname = path.join(args.tablesDir, args.sampleId)
print 'Writing sample table %s' % t_sample_fname
t_sample.write(t_sample_fname, type='ascii', Writer=CommentedHeader, overwrite=True)

