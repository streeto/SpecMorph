'''
Created on 20/06/2014

@author: andre
'''

from .util import logger
from imfit import Imfit

__all__ = ['fit_image', 'model_image']

################################################################################
def fit_image(flux, noise, guess_model, PSF=None,
              mode='LM', insist=False, quiet=False, nproc=None):
    '''
    Doc me!
    '''
    imfit = Imfit(guess_model, PSF, quiet=quiet, nproc=nproc)
    imfit.fit(flux, noise, mode=mode)
    logger.debug('Valid pix: %d | Iterations: %d | pegged: %d | chi2: %f' % \
                 (imfit.nValidPixels, imfit.nIter, imfit.nPegged, imfit.chi2))
    fitted_model = imfit.getModelDescription()
    if not imfit.fitConverged or imfit.nPegged > 0:
        logger.warn('Bad fit: did not converge or pegged parameter.')
        if mode == 'LM' and insist:
            logger.warn('     Retrying using N-M simplex.')
            imfit.fit(flux, noise, mode='NM')
            fitted_model = imfit.getModelDescription()
            if imfit.fitConverged:
                logger.debug('     N-M simplex chi2: %f' % imfit.chi2)
            else:
                logger.warn('     Bad fit: N-M simplex did not converge.')
                logger.debug('     Initial model:\n%s\n\n' % str(guess_model))
    return fitted_model, imfit.fitConverged, imfit.chi2
################################################################################


################################################################################
def model_image(model, shape, PSF, flux_unit, nproc=None):
    imfit = Imfit(model, PSF, quiet=True, nproc=nproc)
    return imfit.getModelImage(shape) * flux_unit
################################################################################


