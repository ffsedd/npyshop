#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import logging

import imghdr

import numpy as np

from pathlib import Path

from send2trash import send2trash

from tkinter import filedialog

import nputils

FILETYPES = ['jpeg', 'bmp', 'png', 'tiff']


class npImage():

    def __init__(self, fpath=None):
        self.fpath = fpath
        self.arr = None
        self.channels = None
        self.bitdepth = None
        self.original = None

        if fpath:
            self.load(fpath)

    def load(self, fpath=None):

        fpath = fpath or filedialog.askopenfilename()
        print(f"open file {fpath}")

        # make sure it's an image file
        self.mode = imghdr.what(fpath)
        self.name = Path(fpath).stem
        self.filesize = Path(fpath).stat().st_size

        assert self.mode in FILETYPES, f"Error, not supported {fpath}"

        self.fpath = fpath

        self.arr = nputils.load_image(fpath)

        self.bitdepth = nputils.get_bitdepth(self.arr)
        self.channels = 1 if self.arr.ndim == 2 else self.arr.shape[2]

        self.arr = nputils.int_to_float(self.arr)  # convert to float
        self.original = self.arr.copy()

        print(f"bitdepth {self.bitdepth}")
        self.info()

    def reset(self):
        self.arr = self.original.copy()

    @property
    def width(self):
        return self.arr.shape[1]

    @property
    def height(self):
        return self.arr.shape[0]

    @property
    def ratio(self):
        return self.arr.shape[0] / self.arr.shape[1]

    def save(self, fpath=None):

        fpath = fpath or self.fpath
        Fp = Path(fpath)
        print(f"save to {Fp} bitdepth:{self.bitdepth} mode:{self.mode}")
        print(f"{self.info()}")

        if Fp.is_file():
            send2trash(str(Fp))

        nputils.save_image(self.arr, fpath, bitdepth=self.bitdepth)

    def save_as(self, fpath=None):
        fpath = fpath or filedialog.asksaveasfilename(defaultextension=".jpg")
        nputils.save_image(self.arr, fpath, bitdepth=self.bitdepth)

    def rotate(self, k=1):
        ''' rotate array by 90 degrees
        k = number of rotations
        '''
        # self.arr = ndimage.rotate(self.arr, angle=-90, reshape=True)
        self.arr = np.rot90(self.arr, k, axes=(0, 1))

    def invert(self):
        self.arr = 1 - self.arr

    def mirror(self):
        self.arr = np.flip(self.arr, 1)

    def flip(self):
        self.arr = np.flip(self.arr, 0)

    def normalize(self):
        self.arr = nputils.normalize(self.arr)

    def gamma(self, g):
        """gamma correction of an numpy float image, where
        gamma = 1. : no effect
        gamma > 1. : image will darken
        gamma < 1. : image will brighten"""
        self.arr = self.arr ** g

    def multiply(self, f):
        """ change brightness """
        self.arr = np.clip(f * self.arr, 0, 1)

    def add(self, f):
        """ change brightness """
        self.arr = np.clip(f + self.arr, 0, 1)

    def contrast(self, f):
        """ change contrast """
        self.arr = np.clip(.5 + f * (self.arr - .5), 0, 1)

    def crop(self, x0, x1, y0, y1):

        # ensure crop area in image
        x0 = int(max(x0, 0))
        x1 = int(min(x1, self.arr.shape[1]))
        y1 = int(min(y1, self.arr.shape[0]))
        y0 = int(max(y0, 0))
        print(f"apply crop: {x0} {x1} {y0} {y1}")
        self.arr = self.arr[y0:y1, x0:x1,...]
        self.info()

    def info(self):
        ''' print info about numpy array'''
        y = self.arr
        if len(y.ravel()) == 0:
            print("array is empty")
        else:
            out = f"{y.dtype}\t{str(y.shape)}\t<{y.min():.3f} \
            {y.mean():.3f} {y.max():.3f}> ({y.std():.3f})\t{type(y)} \
            bitdepth:{self.bitdepth} "
            print(out)
        return out

    @property
    def stats(self):
        ''' return stats dict '''
        return {
            "name": self.name,
            "mode": self.mode,
            "bitdepth": self.bitdepth,
            "channels": self.channels,
            "size": f"{self.filesize/1024/1024: .3f} MB",
            "height": self.height,
            "width": self.width,
            "ratio": round(self.ratio, 2),
            "min": round(self.arr.min(), 2),
            "max": round(self.arr.max(), 2),
            "mean": round(self.arr.mean(), 2),
            "std_dev": round(self.arr.std(), 2),
        }
