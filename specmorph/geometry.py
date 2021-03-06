'''
Created on 18/06/2014

@author: andre
'''

import numpy as np

def gen_rebin(a, e, bin_e, mean=True):
    '''
    Rebinning function. Given the value array `a`, with generic
    positions `e` such that `e.shape == a.shape`, return the sum
    or the mean values of `a` inside the bins defined by `bin_e`.
    
    Parameters
    ----------
    a : array like
        The array values to be rebinned.
    
    e : array like
        Generic positions of the values in `a`.
    
    bin_e : array like
        Bins of `e` to be used in the rebinning.
    
    mean : boolean
        Divide the sum by the number of points inside the bin.
        Defaults to `True`.
        
    Returns
    -------
    a_e : array
        An array of length `len(bin_e)-1`, containing the sum or
        mean values of `a` inside each bin.
        
    Examples
    --------
    
    TODO: Add examples for gen_rebin.
    '''
    if isinstance(a, np.ma.MaskedArray):
        a = a.compressed()
        e = e.compressed()
    else:
        a = a.ravel()
        e = e.ravel()
    a_e = np.histogram(e, bin_e, weights=a)[0]
    if mean:
        N = np.histogram(e, bin_e)[0]
        mask = N > 0
        a_e[mask] /= N[mask]
    return a_e


def r50(X, r, r_max=None):
    '''
    Evaluate radius where the cumulative value of `X` reaches half of its value.

    Parameters
    ----------
    X : array like
        The property whose half radius will be evaluated.
    
    r : array like
        Radius associated to each value of `X`. Must be the
        same shape as `X`.

    r_max : int
        Integrate up to `r_max`. Defaults to `np.max(r)`. 
    
    Returns
    -------
    HXR : float
        The "half X radius."

    Examples
    --------
    
    Find the radius containing half of the volume of a gaussian.
    
    >>> import numpy as np
    >>> xx, yy = np.indices((100, 100))
    >>> x0, y0, A, a = 50.0, 50.0, 1.0, 20.0
    >>> z = A * np.exp(-((xx-x0)**2 + (yy-y0)**2)/a**2)
    >>> r = np.sqrt((xx - 50)**2 + (yy-50)**2)
    >>> getGenHalfRadius(z, r)
    16.786338066912215

    '''
    if r_max is None:
        r_max = r.max()
    bin_r = np.arange(0, r_max, 1)
    cumsum_X = gen_rebin(X, r, bin_r, mean=False).cumsum()

    from scipy.interpolate import interp1d
    invX_func = interp1d(cumsum_X, bin_r[1:])
    halfRadiusPix = invX_func(cumsum_X.max() / 2.0)
    return float(halfRadiusPix)


def ellipse_params(image, x0=0.0, y0=0.0):
    '''
    Estimate ellipticity and position angle of the galaxy using the
    "Stokes parameters", as described in:
    http://adsabs.harvard.edu/abs/2002AJ....123..485S
    The image used is ``qSignal``.
    
    Parameters
    ----------
    image : array
        Image to use when calculating the ellipse parameters.

    x0 : float
        X coordinate of the origin. Defaults to ``0.0``.
    
    y0 : float
        Y coordinate of the origin. Defaults to ``0.0``.
    
    Returns
    -------
    pa : float
        Position angle in degrees, counter-clockwise relative
        to the positive Y axis.
    
    ell : float
        Ellipticity, defined as :math:`1 - b/a`, with :math:`b/a`
        being the ratio between semiminor and semimajor axes.
    '''
    yy, xx = np.indices(image.shape)
    y = yy - y0
    x = xx - x0
    x2 = x**2
    y2 = y**2
    xy = x * y
    r2 = x2 + y2
    
    if isinstance(image, np.ma.MaskedArray): 
        im = image.copy()
    else:
        im = np.ma.array(image, copy=True)
    im[r2 < 0.1] = np.ma.masked
    norm = image.sum()
    Mxx = ((x2 / r2) * im).sum() / norm
    Myy = ((y2 / r2) * im).sum() / norm
    Mxy = ((xy / r2) * im).sum() / norm
    
    Q = Mxx - Myy
    U = Mxy
    
    pa = np.arctan2(U, Q) / 2.0
    
    # b/a ratio
    ba = (np.sin(2 * pa) - U) / (np.sin(2 * pa) + U)
    # Should be the same as ba
    #ba_ = (np.cos(2*pa) - Q) / (np.cos(2*pa) + Q)
    
    # PA is measured degrees from +y axis.
    pa = 90.0 - pa * 180.0 / np.pi
    ell = 1 - ba
    
    return fix_PA_ell(pa, ell)


def distance(shape, x0=0.0, y0=0.0, pa=0.0, ell=0.0):
    '''
    Return an image (:class:`numpy.ndarray`)
    of the distance from the center ``(x0, y0)`` in pixels,
    assuming a projected disk.
    
    Parameters
    ----------
    shape : (float, float)
        Shape of the image to get the pixel distances.
    
    x0 : float, optional
        X coordinate of the origin. Defaults to ``0.0``.
    
    y0 : float, optional
        Y coordinate of the origin. Defaults to ``0.0``.
    
    pa : float, optional
        Position angle in degrees, counter-clockwise relative
        to the positive Y axis.
    
    ell : float, optional
        Ellipticity, defined as :math:`1 - b/a`, with :math:`b/a`
        being the ratio between semiminor and semimajor axes.

    Returns
    -------
    pixelDistance : array
        Pixel distances.

    '''
    y, x = np.indices(shape)
    y = y - y0
    x = x - x0
    x2 = x**2
    y2 = y**2
    xy = x * y

    # Angles here are in radians from the +x axis.
    pa = (90.0 + pa) * np.pi / 180.0
    a_b = 1.0 / (1.0 - ell)
    
    cos_th = np.cos(pa)
    sin_th = np.sin(pa)

    A1 = cos_th ** 2 + a_b ** 2 * sin_th ** 2
    A2 = -2.0 * cos_th * sin_th * (a_b ** 2 - 1.0)
    A3 = sin_th ** 2 + a_b ** 2 * cos_th ** 2

    return np.sqrt(A1 * x2 + A2 * xy + A3 * y2)


def fix_PA_ell(PA, ell):
    '''
    Force position angle (P.A.) into (0, 180) range,
    and ellipticity (:math:``1 - b/a``) to be positive.
    
    Assuming the P.A. describes the orientation of an
    axis-symmetric object, if :math:``P.A. > 180``, then
    :math:``P.A._{fix} = P.A. - 180`` describes the same
    orientation.
    
    When ellipticity is negative, that means the semimajor
    and semiminor axes are swapped. In this case,
    :math:``e_{fix} = -e / (1 - e)``, and  P.A. is rotated
    90 degrees (keeping it in the correct range).
    
    Parameters
    ----------
    PA : float
        Position angle in degrees.
        
    ell : float
        Ellipticity.
    
    Returns
    -------
    PA : float
        Position angle in degrees.
    
    ell : float
        Ellipticity.
    '''
    inv_ell = lambda e: -e / (1 - e)

    PA %= 360.0
    if PA < 0.0:
        PA + 360.0
    elif PA > 180.0:
        PA -= 180.0
    
    if ell < 0.0:
        if PA > 90.0:
            return PA - 90.0, inv_ell(ell)
        else:
            return PA + 90.0, inv_ell(ell)
    else:
        return PA, ell

