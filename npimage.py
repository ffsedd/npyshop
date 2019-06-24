#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import tkinter as tk
from tkinter import filedialog, messagebox


import imghdr



from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np


from matplotlib import pyplot as plt
from skimage import img_as_ubyte, img_as_uint # img_as_float - data normalized !!!

# from scipy import ndimage

from pathlib import Path

from imageio import imread

from numpngw import write_png

from send2trash import send2trash


FILETYPES = ['jpeg', 'bmp', 'png', 'tiff']


def info(y, name="", print_output=True):
    ''' print info about numpy array'''
    if isinstance(y, (np.ndarray, np.generic)):
#        pd.set_option("display.precision", 2)
#        df = pd.DataFrame([[str(name), y.dtype, str(y.shape), y.min(), y.max(), y.mean(), y.std(), type(y) ]],
#                    columns=['name', 'dtype', 'shape', 'min', 'max', 'mean', 'std', 'type' ])
#        print(df.to_string(index=False))
        out = f"{str(name)}\t{y.dtype}\t{str(y.shape)}\t<{y.min():.3f} {y.mean():.3f} {y.max():.3f}> ({y.std():.3f})\t{type(y)} "
        # print(np.info(y))
    else:
        out = f"{name}\t// {type(y)}"
    if print_output:
        print(out)
    return out


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

def save_image(im, fp_out, bitdepth=None, mode=None):
    ''' float or uint16 numpy array --> 16 bit png
        uint8 numpy array --> 8bit png '''

    logging.info(f"saving image {fp_out}")
    print(f"saving image {fp_out} {info(im)}")
    Fp = Path(fp_out)
    Fp.parent.mkdir(exist_ok=True)
    
    im = np.clip(im,a_min=0,a_max=1)

    if Fp.suffix.lower() == ".jpg":
        numpy_to_jpg(im, fp_out)

    elif Fp.suffix.lower() == ".png":
        if bitdepth == 8:
            im = img_as_ubyte(im)
        elif bitdepth == 16: # 16bit PNG - default output
            im = img_as_uint(im)  # accept float
        else:
            raise Exception(f"You must specify bitdepth:{bitdepth}")
        numpy_to_png(im, fp_out,  bitdepth=bitdepth)
    else:
        raise Exception(f"Not implemented file ext:{Fp.suffix}")
    logging.info(f"image saved")

#
def plti(im, name="", plot_axis=False, vmin=0, vmax=1, **kwargs):

    cmap = "gray" if im.ndim == 2 else "jet"
    plt.title(name)
    plt.imshow(im, interpolation="none", cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    if not plot_axis:
        plt.axis('off')  # turn off axis
    plt.show()

def numpy_to_png(im, fp_out,  bitdepth=8):
    
    assert isinstance(im, (np.ndarray, np.generic))
    Fp = Path(fp_out)
    logging.info(f"saving array to png..{fp_out} bitdepth {bitdepth}")
    write_png(Fp.with_suffix(".png"), im, bitdepth=bitdepth)
    logging.info("...saved")


def numpy_to_jpg(im, fp_out):

    assert isinstance(im, (np.ndarray, np.generic))
    Fp = Path(fp_out)
    logging.info(f"saving array to jpg...{fp_out}")
    cmap = "gray" if im.ndim == 2 else "jet"
    plt.imsave(Fp.with_suffix(".jpg"), im, cmap=cmap, vmin=0, vmax=1)  # use matplotlib
    logging.info(f"...saved")

def normalize(y, inrange=None, outrange=(0, 1)):
    ''' Normalize numpy array --> values 0...1 '''

    imin, imax = inrange if inrange else ( np.min(y), np.max(y) )
    omin, omax = outrange
    logging.debug(f"normalize array, limits - in: {imin},{imax} out: {omin},{omax}")

    return np.clip( omin + omax * (y - imin) / (imax - imin), a_min=omin, a_max=omax )






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
          
        #make sure it's an image file
        self.mode = imghdr.what(fpath)
        self.name = Path(fpath).stem
        self.filesize = Path(fpath).stat().st_size
        
        assert self.mode in FILETYPES, f"Error, not supported {fpath}"
        
        
        self.fpath = fpath
        
        self.arr = load_image(fpath)
        
        self.bitdepth = get_bitdepth(self.arr)
        self.channels = 1 if self.arr.ndim == 2 else self.arr.shape[2]
        

        
        self.arr = int_to_float(self.arr) # convert to float
        self.original = self.arr.copy()
        
        print(f"bitdepth {self.bitdepth}")  
        info(self.arr)

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
        
        if Fp.is_file():
            send2trash(str(Fp))
        
        save_image(self.arr, fpath, bitdepth=self.bitdepth)


    def save_as(self, fpath=None):
        fpath = fpath or filedialog.asksaveasfilename(defaultextension=".jpg")
        save_image(self.arr, fpath, bitdepth=self.bitdepth)


    def rotate(self, k=1):
        ''' rotate array by 90 degrees
        k = number of rotations
        '''
        # self.arr = ndimage.rotate(self.arr, angle=-90, reshape=True)
        self.arr = np.rot90(self.arr, k, axes=(0,1))
        
        
    def invert(self):
        self.arr = 1 - self.arr
        
    def mirror(self):
        self.arr = np.flip(self.arr, 1)
        
    def flip(self):
        self.arr = np.flip(self.arr, 0)
        
    def normalize(self):
        self.arr = normalize(self.arr)
        
    def gamma(self, g):
        """gamma correction of an numpy float image, where gamma = 1. : no effect
        gamma > 1. : image will darken
        gamma < 1. : image will brighten"""
        self.arr = self.arr ** g
        


    def crop(self, x0, x1, y0, y1):
               
        # ensure crop area in image
        x0 = int(max(x0, 0))
        x1 = int(min(x1, self.arr.shape[1]))
        y1 = int(min(y1, self.arr.shape[0]))
        y0 = int(max(y0, 0))
        print(f"apply crop: {x0} {x1} {y0} {y1}")
        self.arr = self.arr[y0:y1,x0:x1,:]
        self.info()
        
    def info(self):
        ''' print info about numpy array'''
        y = self.arr
        if len(y.ravel()) == 0:
            print("array is empty")
        else:    
            out = f"{y.dtype}\t{str(y.shape)}\t<{y.min():.3f} {y.mean():.3f} {y.max():.3f}> ({y.std():.3f})\t{type(y)} "
            print(out)
        return out
