'''
Created on 20/06/2014

@author: andre
'''

import logging

logger = logging.getLogger('specmorph')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.ERROR)
