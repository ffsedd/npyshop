#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np

from matplotlib import pyplot as plt

from pathlib import Path

from imageio import imread, imwrite

from skimage import img_as_float, img_as_ubyte, img_as_uint

from testing.timeit import timeit


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
    return imread(fp)


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



