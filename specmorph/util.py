'''
Created on 20/06/2014

@author: andre
'''

import logging
import numpy as np

logger = logging.getLogger('specmorph')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.ERROR)


def find_nearest_index(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx
