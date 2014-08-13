'''
Created on 12/08/2014

@author: andre
'''

from specmorph.util import logger
from tables import Filters
from tables.atom import Atom
import numpy as np

__all__ = ['save_compound_array', 'save_array']

################################################################################
def save_compound_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        logger.warn('Removing existing group %s' % name)
        parent._f_getChild(name)._f_remove(recursive=True)
    grp = db.createGroup(parent, name)
    # HACK: pytables does not support compound dtypes.
    for field in data.dtype.names:
        save_array(db, grp, field, data[field], overwrite)
################################################################################

    
################################################################################
def save_array(db, parent, name, data, overwrite=False):
    if overwrite and name in parent:
        logger.warn('Removing existing array %s' % name)
        parent._f_getChild(name)._f_remove()
    ca = db.createCArray(parent, name, Atom.from_dtype(data.dtype), data.shape, filters=Filters(1, 'blosc'))
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled()
    ca[...] = data
################################################################################
