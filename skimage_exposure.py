#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from warnings import warn



def histogram(image, nbins=256, source_range='image', normalize=False):
    """Return histogram of image.
    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.
    The histogram is computed on the flattened image: for color images, the
    function should be used separately on each channel to obtain a histogram
    for each color channel.
    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    source_range : string, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    See Also
    --------
    cumulative_distribution
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([107432, 154712]), array([0. , 0.5, 1. ]))
    >>> exposure.histogram(image, nbins=2)
    (array([107432, 154712]), array([0.25, 0.75]))
    """
    sh = image.shape

    image = image.flatten()

    hist_range = None

    hist, bin_edges = np.histogram(image, bins=nbins, range=hist_range)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers
    
    
def cumulative_distribution(image, nbins=256):
    """Return cumulative distribution function (cdf) for the given image.
    Parameters
    ----------
    image : array
        Image array.
    nbins : int, optional
        Number of bins for image histogram.
    Returns
    -------
    img_cdf : array
        Values of cumulative distribution function.
    bin_centers : array
        Centers of bins.
    See Also
    --------
    histogram
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cumulative_distribution_function
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> hi = exposure.histogram(image)
    >>> cdf = exposure.cumulative_distribution(image)
    >>> np.alltrue(cdf[0] == np.cumsum(hi[0])/float(image.size))
    True
    """
    hist, bin_centers = histogram(image, nbins)
    img_cdf = hist.cumsum()
    img_cdf = img_cdf / float(img_cdf[-1])
    return img_cdf, bin_centers
