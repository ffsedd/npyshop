#!/usr/bin/env python3
import imghdr
import numpy as np
from pathlib import Path
from send2trash import send2trash
from tkinter import filedialog

import nputils

from testing.timeit import timeit

FILETYPES = ['jpeg', 'bmp', 'png', 'tiff']


class npImage():

    def __init__(self, fpath=None):
        self.fpath = fpath
        self.arr = None
        self.channels = None
        self.bitdepth = None
        self.original = None
        self.slice = np.s_[:, :, ...]

        if fpath:
            self.load(fpath)

    @property
    def properties(self):
        return f"{self.fpath}      |  {self.bitdepth}bit {self.mode}  |  {self.filesize/2**20:.2f} MB  |  {self.width} x {self.height} x {self.channels}  |"


#    @timeit
    def get_mode(self):
        self.mode = imghdr.what(self.fpath)
        assert self.mode in FILETYPES, f"Error, not supported {self.fpath}"


    def load(self, fpath=None):

        fpath = fpath or filedialog.askopenfilename()
        print(f"open file {fpath}")

        # make sure it's an image file
        #
        self.name = Path(fpath).stem
        self.filesize = Path(fpath).stat().st_size
        self.fpath = fpath
        self.get_mode()

        self.arr = nputils.load_image(fpath)

        self.bitdepth = nputils.get_bitdepth(self.arr)
        self.channels = 1 if self.arr.ndim == 2 else self.arr.shape[2]

        self.arr = nputils.int_to_float(self.arr)  # convert to float
        self.original = self.arr.copy()

        print(f"bitdepth {self.bitdepth}")
#        self.info()

    def rgb2gray(self):
        if self.arr.ndim > 2:
            self.arr = nputils.rgb2gray(self.arr)

    def reset(self):
        self.arr = self.original.copy()

    @property
    def center(self):
        x, y = (size//2 for size in self.arr.shape[:2])
        return x, y

    @property
    def width(self):
        return self.arr.shape[1]

    @property
    def height(self):
        return self.arr.shape[0]

    @property
    def ratio(self):
        return self.arr.shape[0] / self.arr.shape[1]

    @timeit
    def save(self, fpath=None):

        fpath = fpath or self.fpath
        Fp = Path(fpath)
        print(f"save to {Fp} bitdepth:{self.bitdepth} mode:{self.mode}")
#        print(f"{self.info()}")

        if Fp.is_file():
            send2trash(str(Fp))

        nputils.save_image(self.arr, fpath, bitdepth=self.bitdepth)
        self.fpath = fpath

    def save_as(self, fpath=None):
        fpath = fpath or filedialog.asksaveasfilename(defaultextension=".jpg")
        nputils.save_image(self.arr, fpath, bitdepth=self.bitdepth)

    def rotate(self, k=1):
        ''' rotate array by 90 degrees
        k = number of rotations
        '''
        # self.arr = ndimage.rotate(self.arr, angle=-90, reshape=True)
        self.arr = np.rot90(self.arr, -k, axes=(0, 1))

    def free_rotate(self, angle):
        ''' rotate array
        '''
        from scipy.ndimage import rotate
        self.arr = rotate(self.arr, angle,
                                  reshape=True, mode='nearest')

        self.info()
        self.arr = np.clip(self.arr, 0, 1)

    def invert(self):
        self.arr[self.slice] = 1 - self.arr[self.slice]

    def mirror(self):
        self.arr[self.slice] = np.flip(self.arr[self.slice], 1)

    def flip(self):
        self.arr[self.slice] = np.flip(self.arr[self.slice], 0)

    def normalize(self):
        self.arr[self.slice] = nputils.normalize(self.arr[self.slice])

    def gamma(self, g):
        """gamma correction of an numpy float image, where
        gamma = 1. : no effect
        gamma > 1. : image will darken
        gamma < 1. : image will brighten"""
        y = self.arr[self.slice] ** g
        self.arr[self.slice] = np.clip(y, 0, 1)

    def unsharp_mask(self, radius, amount):
        from scipy.ndimage import gaussian_filter
        y = self.arr[self.slice]
        mask = gaussian_filter(y, radius)
        y = y + amount * (y - mask)
        self.arr[self.slice] = np.clip(y, 0, 1)

    def blur(self, radius=3):
        from scipy.ndimage import gaussian_filter
        y = self.arr[self.slice]
        y = gaussian_filter(y, radius)
        self.arr[self.slice] = np.clip(y, 0, 1)

    def multiply(self, f):
        """ multiply by scalar and clip """
        y = .5 + f * (self.arr[self.slice] - .5)
        self.arr[self.slice] = np.clip(y, 0, 1)

    def contrast(self, f):
        """ change contrast """
        y = f * self.arr[self.slice]
        self.arr[self.slice] = np.clip(y, 0, 1)

    def fill(self, f=1):
        """ change to constant """
        self.arr[self.slice] = f

    def add(self, f):
        """ change brightness """
        self.arr[self.slice] = np.clip(f + self.arr[self.slice], 0, 1)

    def tres_high(self, f):
        """  """
        self.arr[self.slice][(self.arr[self.slice] > f)] = 1

    def tres_low(self, f):
        """  """
        self.arr[self.slice][(self.arr[self.slice] < f)] = 0

    def clip_high(self, f):
        """  """
        self.arr[self.slice][(self.arr[self.slice] > f)] = f

    def clip_low(self, f):
        """  """
        self.arr[self.slice][(self.arr[self.slice] < f)] = f

    def sigma(self, sigma=2):
        """ s shaped curve """
        y = np.tanh((self.arr[self.slice] - .5) * sigma) / 2 + .5
        self.arr[self.slice] = np.clip(y, 0, 1)

    def crop(self, x0, y0, x1, y1):

        # ensure crop area in image
        #        x0 = int(max(x0, 0))
        #        x1 = int(min(x1, self.arr.shape[1]))
        #        y1 = int(min(y1, self.arr.shape[0]))
        #        y0 = int(max(y0, 0))
        print(f"apply crop: {x0} {x1} {y0} {y1}")
        self.arr = self.arr[self.slice]
#        self.info() # slow

    def fft(self):
        from scipy import fftpack
        # Take the fourier transform of the image.
        y = self.arr * 255
        nputils.info(y)
        F1 = fftpack.fft2(y)
        nputils.info(F1)
        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = np.fft.fftshift(F1).real
        nputils.info(F2)
        ftimage = np.abs(F2)
        ftimage = np.log(ftimage)
        nputils.info(ftimage)
        return ftimage

    def ifft(self):
        nputils.info(self.arr)
        F2 = np.exp(self.arr)
        nputils.info(F2)
        F1 = np.fft.ifftshift(F2)
        img = np.fft.ifft2(F1).real
        nputils.info(img)
        img = img / 255
        nputils.info(img)

        return img

    def high_pass(self, sigma):
        from scipy.ndimage import gaussian_filter
        y = self.arr[self.slice]
        bg = gaussian_filter(y, sigma=sigma)
        y -= bg
        np.clip(y, 0, 1)  # inplace
        self.arr[self.slice] = y

    def info(self):
        ''' print info about numpy array
        very slow with large images '''
        y = self.arr[self.slice]
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
        ''' return stats dict
        statistics very slow with large images - disabled
        '''
        return {
            "name": self.name,
            "mode": self.mode,
            "bitdepth": self.bitdepth,
            "channels": self.channels,
            "size": f"{self.filesize/1024/1024: .3f} MB",
            "height": self.height,
            "width": self.width,
            "ratio": round(self.ratio, 2),
            #            "min": round(self.arr[self.slice].min(), 2),
            #            "max": round(self.arr[self.slice].max(), 2),
            #            "mean": round(self.arr[self.slice].mean(), 2),
            #            "std_dev": round(self.arr[self.slice].std(), 2),
        }

#    @timeit
    def histogram_data(self, bins=256):
        ''' return dict of histogram values (1D)
        result looks like: (0,10,20...), {"red":(15, 7, 3...) ...}

        '''

        colors = ('red', 'green', 'blue', 'black')
#        x = np.linspace(0, 2 ** self.bitdepth - 1, HISTOGRAM_BINS)
        x = np.linspace(0, 1, bins)

        # reset data
        hist_data = {color:x*0 for color in colors}



        if self.arr.ndim > 2:  # rgb or rgba
            for i, color in enumerate(colors[:3]):
                channel = self.arr[self.slice][i]
                hist_data[color] = np.histogram(channel, bins=bins,
                         range=(0, 1), density=True)[0]


        else:   # k or ka
            channel = self.arr[self.slice]
            hist_data['black'] = np.histogram(channel, bins=bins,
                         range=(0, 1), density=True)[0]

        return x, hist_data
