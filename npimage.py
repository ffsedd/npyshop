#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import tkinter as tk
from tkinter import filedialog, messagebox


import imghdr

from collections import deque



from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import numpy as np


from matplotlib import pyplot as plt
from skimage import img_as_float, img_as_ubyte, img_as_uint

from scipy import ndimage

from pathlib import Path

from imageio import imread

from numpngw import write_png


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
    # ensure extenion
    if not Path(fp_out).suffix == ".png":
        fp_out += ".png"

    logging.info(f"saving array to png..{fp_out} bitdepth {bitdepth}")
    #Image.fromarray(img_as_ubyte(im)).save(fp_out)
    write_png(fp_out, im, bitdepth=bitdepth)
    logging.info("...saved")


def numpy_to_jpg(im, fp_out):

    im = img_as_float(im)
    Fp = Path(fp_out)

    logging.info(f"saving array to jpg...{fp_out}")
    
    #Image.fromarray(img_as_ubyte(im)).save(fp_out)  # use PIL
    cmap = "gray" if im.ndim == 2 else "jet"

    plt.imsave(Fp.with_suffix(".jpg"), im, cmap=cmap, vmin=0, vmax=1)  # use matplotlib
    
    logging.info(f"...saved")







class npImage():
    

    def __init__(self, fpath=None):
        self.fpath = fpath
        self.arr = None
        self.channels = None
        self.bitdepth = None
        
        if fpath:
            self.load(fpath)

    def load(self, fpath=None):
        
        fpath = fpath or filedialog.askopenfilename()
        print(f"open file {fpath}")  
          
        #make sure it's an image file
        self.mode = imghdr.what(fpath)
        
        assert self.mode in FILETYPES, f"Error, not supported {fpath}"
        
        
        self.fpath = fpath
        
        self.arr = load_image(fpath)
        
        self.bitdepth = get_bitdepth(self.arr)
        self.channels = 1 if self.arr.ndim == 2 else self.arr.shape[2]
        
        self.arr = img_as_float(self.arr) # convert to float
        
        
        print(f"bitdepth {self.bitdepth}")  
        info(self.arr)
        


    def save(self, fpath=None):
        
        fpath = fpath or self.fpath
        print(f"save to {fpath} bitdepth:{self.bitdepth} mode:{self.mode}")
        save_image(self.arr, fpath, bitdepth=self.bitdepth)


    def save_as(self, fpath=None):
        fpath = fpath or filedialog.asksaveasfilename(defaultextension=".jpg")
        save_image(self.arr, fpath, bitdepth=self.bitdepth)


    def rotate(self):
        print("rotate")
        self.arr = ndimage.rotate(self.arr, angle=-90, reshape=True)
        
        
        
    def invert(self):
        print("invert")
        self.arr = 1 - self.arr
        
    def gamma(self, g):
        """gamma correction of an numpy float image, where gamma = 1. : no effect
        gamma > 1. : image will darken
        gamma < 1. : image will brighten"""
        print("gamma", g)
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
