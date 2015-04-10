'''
Created on 12/03/2015

@author: andre
'''

import atpy
import numpy as np
from os import path

__all__ = ['califa_id_from_cube', 'califa_id_to_str', 'califa_id_to_int', 'load_morph_class']


#############################################################################
def califa_id_from_cube(f):
    '''
    Return the CALIFA ID (as string) from a PyCASSO datacube filename.
    '''
    base = path.basename(f)
    return base[0:5]
#############################################################################


#############################################################################
def califa_id_to_str(califa_id):
    '''
    Convert the CALIFA ID from integer to string (KNNNN).
    '''
    return 'K%04d' % califa_id
#############################################################################


#############################################################################
def califa_id_to_int(califa_id):
    '''
    Convert the CALIFA ID from string (KNNNN) to int.
    '''
    return int(califa_id[1:])
#############################################################################


#############################################################################
def load_masterlist(fname):
    from califa import masterlist  # @UnusedImport
    
    t = atpy.Table(fname, type='califa_masterlist')
    
    # remove last item (Mice b)
    return t.rows(np.arange(len(t) - 1))
#############################################################################


#############################################################################
def load_morph_class(fname):
    '''
    Load the morphology classification table from CALIFA
    ancillary data achive. Converts the Hubble types to
    a numerical sequence (using the prescription by
    C. J. Walcher) and reorganizes the columns.
    
    Returns an ATpy table.
    '''
    t = atpy.Table(fname)
    
    #############################################################################
    # Prescription from the table header by C. J. Walcher.
    #############################################################################
    
    t.add_column('type', np.ones(len(t), dtype='int') * -1)
    
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '0')] = 0
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '1')] = 1
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '2')] = 2
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '3')] = 3
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '4')] = 4
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '5')] = 5
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '6')] = 6
    t.type[(t.hubtyp == 'E') & (t.hubsubtyp == '7')] = 7
    
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == '0')] = 8
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == '0a')] = 9
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'a')] = 10
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'ab')] = 11
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'b')] = 12
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'bc')] = 13
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'c')] = 14
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'cd')] = 15
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'd')] = 16
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'dm')] = 17
    t.type[(t.hubtyp == 'S') & (t.hubsubtyp == 'm')] = 18
    
    t.type[t.hubtyp == 'I'] = 19
    
    assert (t.type != -1).all()
    
    t.add_column('barred', np.ones(len(t), dtype='int') * -1)
    
    t.barred[t.bar == 'A'] = 0
    t.barred[t.bar == 'AB'] = 1
    t.barred[t.bar == 'B'] = 2
    
    assert (t.barred != -1).all()
    
    t.add_column('merger', np.ones(len(t), dtype='int') * -1)
    
    t.merger[t.merg == 'I'] = 0
    t.merger[t.merg == 'M'] = 1
    
    assert (t.barred != -1).all()
    
    t.add_column('hubble_type', [typ + subt for typ, subt in zip(t.hubtyp, t.hubsubtyp)], dtype='S03')
    t.remove_columns(['hubtyp', 'hubsubtyp', 'bar', 'merg'])
    
    #############################################################################
    
    t.add_column('type_min', np.ones(len(t), dtype='int') * -1)
    
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '0')] = 0
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '1')] = 1
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '2')] = 2
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '3')] = 3
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '4')] = 4
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '5')] = 5
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '6')] = 6
    t.type_min[(t.minhubtyp == 'E') & (t.minhubsubtyp == '7')] = 7
    
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == '0')] = 8
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == '0a')] = 9
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'a')] = 10
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'ab')] = 11
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'b')] = 12
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'bc')] = 13
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'c')] = 14
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'cd')] = 15
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'd')] = 16
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'dm')] = 17
    t.type_min[(t.minhubtyp == 'S') & (t.minhubsubtyp == 'm')] = 18
    
    t.type_min[t.minhubtyp == 'I'] = 19
    
    assert (t.type_min != -1).all()
    
    t.add_column('barred_min', np.ones(len(t), dtype='int') * -1)
    
    t.barred_min[t.minbar == 'A'] = 0
    t.barred_min[t.minbar == 'AB'] = 1
    t.barred_min[t.minbar == 'B'] = 2
    
    assert (t.barred_min != -1).all()
    
    t.add_column('merger_min', np.ones(len(t), dtype='int') * -1)
    
    t.merger_min[t.minmerg == 'I'] = 0
    t.merger_min[t.minmerg == 'M'] = 1
    
    assert (t.merger_min != -1).all()
    
    t.add_column('hubble_type_min', [typ + subt for typ, subt in zip(t.minhubtyp, t.minhubsubtyp)], dtype='S03')
    t.remove_columns(['minhubtyp', 'minhubsubtyp', 'minbar', 'minmerg'])
    
    #############################################################################
    
    t.add_column('type_max', np.ones(len(t), dtype='int') * -1)
    
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '0')] = 0
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '1')] = 1
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '2')] = 2
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '3')] = 3
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '4')] = 4
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '5')] = 5
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '6')] = 6
    t.type_max[(t.maxhubtyp == 'E') & (t.maxhubsubtyp == '7')] = 7
    
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == '0')] = 8
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == '0a')] = 9
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'a')] = 10
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'ab')] = 11
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'b')] = 12
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'bc')] = 13
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'c')] = 14
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'cd')] = 15
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'd')] = 16
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'dm')] = 17
    t.type_max[(t.maxhubtyp == 'S') & (t.maxhubsubtyp == 'm')] = 18
    
    t.type_max[t.maxhubtyp == 'I'] = 19
    
    assert (t.type_max != -1).all()
    
    t.add_column('barred_max', np.ones(len(t), dtype='int') * -1)
    
    t.barred_max[t.maxbar == 'A'] = 0
    t.barred_max[t.maxbar == 'AB'] = 1
    t.barred_max[t.maxbar == 'B'] = 2
    
    assert (t.barred_max != -1).all()
    
    t.add_column('merger_max', np.ones(len(t), dtype='int') * -1)
    
    t.merger_max[t.maxmerg == 'I'] = 0
    t.merger_max[t.maxmerg == 'M'] = 1
    
    assert (t.merger_max != -1).all()
    
    t.add_column('hubble_type_max', [typ + subt for typ, subt in zip(t.maxhubtyp, t.maxhubsubtyp)], dtype='S03')
    t.remove_columns(['maxhubtyp', 'maxhubsubtyp', 'maxbar', 'maxmerg'])

    return t

