#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

import numpy as np

from matplotlib import pyplot as plt
from skimage import img_as_ubyte, img_as_uint
# img_as_float - data normalized !!!

from pathlib import Path

from imageio import imread, imwrite







def int_to_float(arr):

    if arr.dtype == np.uint8:
        y = arr / 255
    elif arr.dtype == np.uint16:
        y = arr / 65535
    return y.astype(np.float32)

def load_image(fp):
    ''' load image from fp and return numpy uint8 or uint16 '''
    return imread(fp)

def get_bitdepth(arr):

    if arr.dtype == np.uint8:
        return 8
    elif arr.dtype == np.uint16:
        return 16


def normalize(y, inrange=None, outrange=(0, 1)):
    ''' Normalize numpy array --> values 0...1 '''

    imin, imax = inrange if inrange else ( np.min(y), np.max(y) )
    omin, omax = outrange
    logging.debug(f"normalize array, limits - in: {imin},{imax} out: {omin},{omax}")

    return np.clip( omin + omax * (y - imin) / (imax - imin), a_min=omin, a_max=omax )



def save_image(im, fp_out, bitdepth=None, mode=None):
    ''' float or uint16 numpy array --> 16 bit png
        uint8 numpy array --> 8bit png '''

    logging.debug(f"saving image {fp_out}")
    Fp = Path(fp_out)
    Fp.parent.mkdir(exist_ok=True)

    assert isinstance(im, (np.ndarray, np.generic))
    im = np.clip(im,a_min=0,a_max=1)

    # convert float to int
    if bitdepth == 8:
        im = img_as_ubyte(im)
    elif bitdepth == 16: # 16bit PNG - default output
        im = img_as_uint(im)  # accept float
    else:
        raise Exception(f"You must specify bitdepth:{bitdepth}")
    imwrite(Fp, im)

    logging.debug(f"image saved")

def plti(im, name="", plot_axis=False, vmin=0, vmax=1, **kwargs):

    cmap = "gray" if im.ndim == 2 else "jet"
    plt.title(name)
    plt.imshow(im, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if not plot_axis:
        plt.axis('off')  # turn off axis
    plt.show()




