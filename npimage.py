#!/usr/bin/env python3
import imghdr
import numpy as np
import logging
from pathlib import Path
from send2trash import send2trash
from tkinter import filedialog
#from matplotlib.colors import rgb_to_hsv, hsv_to_rgb  # normalizes
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import nputils

from testing.timeit import timeit

FILETYPES = ['jpeg', 'bmp', 'png', 'tiff']


class npImage():

    def __init__(self, fpath=None, arr=None):
        self.fpath = fpath
        self.arr = arr
        self.bitdepth = None
        self.original = None
        self.slice = np.s_[:, :, ...]
        self.filetype = None
        self.filesize = 0


        if fpath:
            self.load(fpath)
            

        logging.info(f"bitdepth {self.bitdepth}")
#        self.info()           

    
    @timeit
    def properties(self):
        return f"{self.fpath}      |  {self.bitdepth}bit {self.filetype}  |  {self.filesize/2**20:.2f} MB  |  {self.width} x {self.height} x {self.channels}  | color:{self.color_model}"

    def __repr__(self):
        return self.properties()
    
    @timeit
    def check_filetype(self):
        filetype = imghdr.what(self.fpath)
        assert filetype in FILETYPES, f"Error, not supported filetype: {filetype} - {self.fpath}"
        return filetype

    @property
    def channels(self):
        return 1 if self.arr.ndim == 2 else self.arr.shape[2]
        
    @property
    def color_model(self):
        return self.__dict__['color_model']

    @color_model.setter
    def color_model(self, model):

        if model == self.color_model:  # do not change anything
            return
        elif model == 'rgb' and self.color_model == 'hsv':  # HSV -> RGB
            self.arr = hsv_to_rgb(self.arr)
        elif model == 'hsv' and self.color_model == 'rgb':  # RGB -> HSV
            # print("rgb max",self.arr.max())
            self.arr = rgb_to_hsv(self.arr)
            # print("v max",self.arr[:,:,2].max())
        elif model == 'gray':  # RGB/HSV -> GRAY
            self.color_model = 'hsv'  # convert to hsv 
            self.arr = self.arr[:,:,2] # use value only
        elif model == 'rgb' and self.color_model == 'gray':  # GRAY -> RGB 
            self.arr = np.stack((self.arr)*3, axis=-1)
        elif model == 'hsv' and self.color_model == 'gray':  # GRAY -> HSV 
            self.arr = np.stack((np.zeros_like(self.arr))*2, self.arr, axis=-1)
            
        self.__dict__['color_model'] = model  # conversion done, update mode

    @timeit
    def load(self, fpath=None):
        if not fpath:
            logging.info("fpath input dialog")
            fpath = filedialog.askopenfilename()
        if not fpath:
            return

        logging.info(f"open file {fpath}")
        
        Fpath = Path(fpath)



        # make sure it's an image file
        #
        self.name = Fpath.stem
        self.filesize = Fpath.stat().st_size
        self.fpath = Fpath
        self.filetype = self.check_filetype()

        self.arr = nputils.load_image(Fpath)
        self.bitdepth = nputils.get_bitdepth(self.arr)
        self.__dict__['color_model'] = 'rgb' if self.channels == 3 else 'gray' 
        
        self.arr = nputils.int_to_float(self.arr)  # convert to float
        # self.original = self.arr.copy()
        
    def get_selection(self):
        return self.arr[self.slice]
        
    def set_selection(self,  y):
        self.arr[self.slice] = y
        
    @timeit
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
        logging.info(f"save to {Fp} bitdepth:{self.bitdepth} filetype:{self.filetype}")
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
    
    @timeit
    def free_rotate(self, angle):
        ''' rotate array
        '''
        from scipy.ndimage import rotate
        self.arr = rotate(self.arr, angle,
                                  reshape=True, mode='nearest')

        self.info()
        self.arr = np.clip(self.arr, 0, 1)


    def crop(self, x0, y0, x1, y1):

        # ensure crop area in image
        #        x0 = int(max(x0, 0))
        #        x1 = int(min(x1, self.arr.shape[1]))
        #        y1 = int(min(y1, self.arr.shape[0]))
        #        y0 = int(max(y0, 0))
        logging.info(f"apply crop: {x0} {x1} {y0} {y1}")
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


    def info(self):
        ''' print info about numpy array
        very slow with large images '''
        y = self.arr
        # if len(y.ravel()) == 0:
            # print("array is empty")
        # else:
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
            "filetype": self.filetype,
            "bitdepth": self.bitdepth,
            "channels": self.channels,
            "size": f"{self.filesize/1024/1024: .3f} MB",
            "height": self.height,
            "width": self.width,
            "ratio": round(self.ratio, 2),
            "min": round(self.arr[self.slice].min(), 2),
            "max": round(self.arr[self.slice].max(), 2),
            "mean": round(self.arr[self.slice].mean(), 2),
            "std_dev": round(self.arr[self.slice].std(), 2),
        }
