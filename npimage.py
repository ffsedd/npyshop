#!/usr/bin/env python3
import imghdr
import numpy as np
import logging
from pathlib import Path
from send2trash import send2trash
from tkinter import filedialog
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from qq.skimage_dtype import img_as_float, img_as_ubyte, img_as_uint
from imageio import imread, imwrite

FILETYPES = ['jpeg', 'bmp', 'png', 'tiff']


class npImage():

    def __init__(self, img_path=None, img_arr=None, fft=None):
        self.fpath = img_path
        self.arr = img_arr
        self.bitdepth = None
        self.original = None
        self.slice = np.s_[:, :, ...]
        self.filetype = None
        self.filesize = 0
        self.color_model = 'gray'
        self.fft = None

        if img_path:
            self.load(img_path)


        logging.info(f"bitdepth {self.bitdepth}")
#        self.info()



    def properties(self):
        return f"{self.fpath}      |  {self.bitdepth}bit {self.filetype}  |  {self.filesize/2**20:.2f} MB  |  {self.width} x {self.height} x {self.channels}  | color:{self.color_model}"

    def __repr__(self):
        return self.properties()


    def check_filetype(self):
        filetype = imghdr.what(self.fpath)
        assert filetype in FILETYPES, f"Error, not supported filetype: {filetype} - {self.fpath}"
        return filetype

    @property
    def channels(self):
        return 1 if self.arr.ndim == 2 else self.arr.shape[2]


    def color_model_change(self, model):
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

        self.color_model = model  # conversion done, update mode


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

        self.arr = imread(Fpath)
        self.bitdepth = self._get_bitdepth(self.arr) # orig bitdepth before conversion to float
        self.color_model = 'rgb' if self.channels == 3 else 'gray'

        self.arr = img_as_float(self.arr)  # convert to float
        # self.original = self.arr.copy()



    def _get_bitdepth(self, arr):
        ''' read bitdepth before conversion to float '''
        if arr.dtype == np.uint8:
            return 8
        elif arr.dtype == np.uint16:
            return 16
        else:
            raise Exception(f"unsupported array type: {arr.dtype}")


    def get_selection(self):
        return self.arr[self.slice]

    def set_selection(self,  y):
        self.arr[self.slice] = y


    def rgb2gray(self):
        if self.arr.ndim > 2:
            self.arr = np.dot(self.arr[..., :3], [0.2989, 0.5870, 0.1140])

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


    def save(self, fpath=None):

        fpath = fpath or self.fpath
        Fp = Path(fpath)
        logging.info(f"save to {Fp} bitdepth:{self.bitdepth} filetype:{self.filetype}")
        print(f"save image {fpath}: {self.info()}")

        if Fp.is_file():
            try:
                send2trash(str(Fp))
            except Exception as e:
                logging.info(f"send2trash failed {Fp}")
                Fp.unlink()
                
        self._save_image(self.arr, fpath=fpath, bitdepth=self.bitdepth)
        self.fpath = fpath

    def _save_image(self, float_arr, fpath, bitdepth=8):
        ''' '''

        assert isinstance(float_arr, (np.ndarray, np.generic))
    
        Fp = Path(fpath)
        Fp.parent.mkdir(exist_ok=True)

        float_arr = np.clip(float_arr, a_min=0, a_max=1)
    
        arr = self._float_to_int(float_arr, bitdepth)
    
        imwrite(Fp, arr)
    
        logging.debug(f"image saved")

    def _float_to_int(self, arr, bitdepth=8):
        ''' '''
        if bitdepth == 8:
            return img_as_ubyte(arr)
        elif bitdepth == 16:
            return img_as_uint(arr)
        else:
            raise Exception("unsupported bitdepth")


    def save_as(self, fpath=None):
        fpath = fpath or filedialog.asksaveasfilename(defaultextension=".jpg")
        self._save_image(self.arr, fpath, bitdepth=self.bitdepth)

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


    def crop(self, x0, y0, x1, y1):

        # ensure crop area in image
        #        x0 = int(max(x0, 0))
        #        x1 = int(min(x1, self.arr.shape[1]))
        #        y1 = int(min(y1, self.arr.shape[0]))
        #        y0 = int(max(y0, 0))
        logging.info(f"apply crop: {x0} {x1} {y0} {y1}")
        self.arr = self.arr[self.slice]
#        self.info() # slow



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



    def make_fft(self):
        from scipy import fftpack
        # Take the fourier transform of the image.
        y = 255 * self.arr
        F1 = fftpack.fft2(y)
        # Now shift the quadrants around so that low spatial frequencies are in
        # the center of the 2D fourier transformed image.
        F2 = fftpack.fftshift(F1)

        y = 20 * np.log10(np.abs(F2.real) + .1)
        y /= 255
        self.fft = F2
        self.arr = y
        logging.info("fft created")


    def make_ifft(self):
        if self.fft is None:
            logging.info("no fft image")
            return

        from scipy import fftpack
        F2 = self.fft
        F1 = fftpack.ifftshift(F2)
        y = fftpack.ifft2(F1).real
        y = normalize(y)
        self.arr = y
        self.fft = None

    def fft_toggle(self):
        if self.fft is None:
            self.make_fft()
        else:
            self.make_ifft()




# NUMPY TOOLS ====================================================


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


def normalize(y, inrange=None, outrange=(0, 1)):
    ''' Normalize numpy array --> values 0...1 '''

    imin, imax = inrange if inrange else ( np.min(y), np.max(y) )
    omin, omax = outrange
    logging.debug(f"normalize array, limits - in: {imin},{imax} out: {omin},{omax}")

    return np.clip( omin + omax * (y - imin) / (imax - imin), a_min=omin, a_max=omax )


def np_to_pil(im):
    from PIL import Image
    ''' np float image (0..1) -> 8bit PIL image '''
    return Image.fromarray(img_as_ubyte(im))


def pil_to_np(im):
    ''' PIL image -> np float image (0..1) '''
    return img_as_float(im)



def blur(y, radius):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(y, radius) 

def gray(y):
    if y.ndim == 2:
        return y
    elif y.ndim >= 3:
        return np.dot(y[..., :3], [0.2989, 0.5870, 0.1140])
    else:
        raise Exception(f"gray conversion not supported, array ndim {y.ndim}")
            
       
