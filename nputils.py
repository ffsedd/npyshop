#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np

from matplotlib import pyplot as plt

from pathlib import Path

from imageio import imread, imwrite

from skimage import img_as_float, img_as_ubyte, img_as_uint
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from testing.timeit import timeit


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


@timeit
def int_to_float(arr):
    ''' twice as fast then (arr / 255).astype(np.float) '''
    return img_as_float(arr)


@timeit
def float_to_int(arr, bitdepth):
    ''' '''
    if bitdepth == 8:
        return img_as_ubyte(arr)
    else:
        return img_as_uint(arr)


@timeit
def load_image(fp):
    ''' load image from fp and return numpy uint8 or uint16 '''
    return imread(fp)  # imageio faster
#    return plt.imread(fp)  # matplotlib slower


def get_bitdepth(arr):
    ''' read bitdepth before conversion to float '''
    if arr.dtype == np.uint8:
        return 8
    elif arr.dtype == np.uint16:
        return 16


def normalize(y, inrange=None, outrange=(0, 1)):
    ''' Normalize numpy array --> values 0...1 '''

    imin, imax = inrange if inrange else (np.min(y), np.max(y))
    omin, omax = outrange
    logging.debug(f"normalize array, limits - in: \
        {imin}, {imax} out: {omin}, {omax}")

    return np.clip(omin + omax * (y - imin) / (imax - imin),
                   a_min=omin, a_max=omax)


@timeit
def save_image(float_arr, fp_out, bitdepth=8):
    ''' '''

    logging.debug(f"saving image {fp_out}")
    assert isinstance(float_arr, (np.ndarray, np.generic))

    Fp = Path(fp_out)
    Fp.parent.mkdir(exist_ok=True)
    logging.info(f"save image {Fp.name}")
    float_arr = np.clip(float_arr, a_min=0, a_max=1)

    arr = float_to_int(float_arr, bitdepth)

    imwrite(Fp, arr)

    logging.debug(f"image saved")


def plti(im, name="", plot_axis=False, vmin=0, vmax=1, **kwargs):

    cmap = "gray" if im.ndim == 2 else "jet"
    plt.title(name)
    plt.imshow(im, interpolation="none", cmap=cmap, vmin=vmin,
               vmax=vmax, **kwargs)
    if not plot_axis:
        plt.axis('off')  # turn off axis
    plt.show()


def info(y):
    logging.info(f"{y.dtype}\t{str(y.shape)}\t<{y.min():.3f} \
            {y.mean():.3f} {y.max():.3f}> ({y.std():.3f})\t{type(y)} ")


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask






# MAKE FILTERS WORK WITH RGB

# A decorator is a function that expects ANOTHER function as parameter
def work_with_hsv_decorator(decorated_f):

    def wrapper_f(arr, *args, **kwargs):
        logging.info("convert to hsv")
        arr = rgb_to_hsv(arr)
        y = decorated_f(arr=arr, *args, **kwargs)
        y = hsv_to_rgb(y)
        logging.info("convert to rgb")
        return y
    return wrapper_f

@work_with_hsv_decorator
def gamma(arr, g):
    """gamma correction of an numpy float image, where
    gamma = 1. : no effect
    gamma > 1. : image will darken
    gamma < 1. : image will brighten"""
    arr = arr ** g
    arr = np.clip(arr, 0, 1)
    return arr
