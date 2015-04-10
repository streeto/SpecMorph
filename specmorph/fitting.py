'''
Created on 20/06/2014

@author: andre
'''

from .util import logger
from imfit import Imfit

__all__ = ['fit_image', 'model_image']

################################################################################
def fit_image(flux, noise, guess_model, PSF=None,
              mode='LM', insist=False, quiet=True, nproc=None, use_cash_statistics=False):
    '''
    Doc me!
    '''
    imfit = Imfit(guess_model, PSF, quiet=quiet, nproc=nproc)
    imfit.fit(flux, noise,
              mode=mode,
              error_type='sigma',
              use_cash_statistics=(mode != 'LM' and use_cash_statistics))
    logger.debug('Valid pix: %d | Iter.: %d | pegged: %d | fit stat.: %f | BIC: %f' % \
                 (imfit.nValidPixels, imfit.nIter, imfit.nPegged, imfit.reducedFitStatistic, imfit.BIC))
    fitted_model = imfit.getModelDescription()
    if not imfit.fitConverged or imfit.nPegged > 0:
        logger.warn('Bad fit: did not converge or pegged parameter.')
        if mode == 'LM' and insist:
            logger.warn('     Retrying using N-M simplex.')
            imfit.fit(flux, noise,
                      mode='NM',
                      error_type='sigma',
                      use_cash_statistics=use_cash_statistics)
            fitted_model = imfit.getModelDescription()
            if imfit.fitConverged:
                logger.debug('     N-M simplex fit statistic: %f | BIC: %f' % (imfit.reducedFitStatistic, imfit.BIC))
            else:
                logger.warn('     Bad fit: N-M simplex did not converge.')
                logger.debug('     Initial model:\n%s\n\n' % str(guess_model))
    return fitted_model, imfit.fitConverged, imfit.reducedFitStatistic
################################################################################


################################################################################
def model_image(model, shape, PSF=None, nproc=None):
    imfit = Imfit(model, PSF, quiet=True, nproc=nproc)
    return imfit.getModelImage(shape)
################################################################################


