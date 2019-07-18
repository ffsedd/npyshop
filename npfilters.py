#!/usr/bin/env python3
import numpy as np
import nputils

def clip_result(func):
    ''' decorator to ensure result in limits 0..1 '''
    def wrapper(*args, **kwargs):
        y = func(*args, **kwargs)
        y = np.clip(y, 0, 1)
        return y
    return wrapper


def invert(y):
    return 1 - y


def mirror(y):
    return np.flip(y, 1)


def flip(y):
    return np.flip(y, 0)


def normalize(y):
    ''' Normalize array --> values 0...1 '''
    return (y - y.min()) / y.ptp()


def equalize(y):
    from skimage import exposure
    return exposure.equalize_hist(y)


def adaptive_equalize(y, clip_limit=0.03):
    from skimage import exposure
    return exposure.equalize_adapthist(y, clip_limit=clip_limit)


def gamma(y, g):
    """gamma correction of an numpy float image, where
    g = 1 ~ no effect, g > 1 ~ darken, g < 1 ~ brighten
    """
    return y ** g


def unsharp_mask(y, radius, amount):
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(y, radius)
    y = y + amount * (y - mask)
    return y


def blur(y, radius=3):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(y, radius)


def contrast(y, f):
    """ change contrast """
    return .5 + f * (y - .5)


def multiply(y, f):
    """ multiply by scalar """
    return y * f


def fill(y, f=0):
    """ change to constant """
    return f


def add(y, f):
    """ change brightness """
    return y + f


def tres_high(y, f):
    """  change value of light pixels to 1 """
    y[y > f] = 1
    return y


def tres_low(y, f):
    """ change value of dark pixels to 0 """
    y[y < f] = 0
    return y


def clip_high(y, f):
    """ change value of light pixels to limit """
    y[y > f] = f
    return y


def clip_low(y, f):
    """ change value of dark pixels to limit """
    y[y < f] = f
    return y


def sigma(y, sigma=2):
    """ s shaped curve """
    y = np.tanh((y - .5) * sigma) / 2 + .5
    return y







def high_pass(y, sigma):
    from scipy.ndimage import gaussian_filter
    bg = gaussian_filter(y, sigma=sigma)
    y = y - bg
    return y
