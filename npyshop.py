#!/usr/bin/env python3
import logging
import sys
import time
import os

import tkinter as tk
import numpy as np
from pathlib import Path
from functools import wraps

from PIL import Image, ImageTk

from skimage_dtype import img_as_ubyte

import npimage
import nphistory
import npfilters
import nphistwin
import npstatswin
from npgui import askfloat
from tkinter import filedialog
from npfilelist import FileList
from testing.timeit import timeit

time0 = time.time()
print("imports done")

'''
RESOURCES:
 * Image operations:
      https://homepages.inf.ed.ac.uk/rbf/HIPR2/wksheets.htm
      https://web.cs.wpi.edu/~emmanuel/courses/cs545/S14/slides/lecture02.pdf
 * Fourier transform
      https://www.cs.unm.edu/~brayer/vision/fourier.html
      http://www.imagemagick.org/Usage/fourier/

BUGS:


TODO:


Menu items have an accelerator attribute specifically for this purpose:

accelerator Specifies a string to display at the right side of the menu entry. Normally describes an accelerator keystroke sequence that may be typed to invoke the same function as the menu entry. This option is not available for separator or tear-off entries.

self.file_menu.add_command(..., accelerator="Ctrl+S")



move command functions inside app class? is it possible to decorate them?




    load menu items from csv?

    underline _File _View ...

    command parameters to history

    command history to text file or iptc?

    repeat command

    circular selection


'''

LOG_FPATH = Path(__file__).with_suffix(".log")

CFG = {
    "hide_histogram": True,
    "hide_toolbar": False,
    "hide_stats": True,
    "histogram_bins": 256,
    "history_steps": 10,     # memory !!!
    "image_extensions" : [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"],

}


def commands_dict():
    ''' can not be set as a global, contains undefined functions '''
    return {
        "File":
            [
                ("Open", "o", load),
                ("Save", "S", save),
                ("Save as", "s", save_as),
                ("Save as png", "P", save_as_png),
                ("Previous", "Left", load_previous),
                ("Next", "Right", load_next),
                ("Next", "Up", load_first),
                ("Next", "Down", load_last),

            ],
        "History":
            [
                ("Undo", "z", undo),
                ("Redo", "y", redo),
                ("Original_toggle", "q", toggle_original),

            ],
        "Image":
            [
                ("Crop", "C", crop),
                ("Rotate_90", "r", rotate_90),
                ("Rotate_270", "R", rotate_270),
                ("Rotate_180", "u", rotate_180),
                ("Free Rotate", "f", free_rotate),
                ("rgb2gray", "b", rgb2gray),
                ("FFT toggle", "F", fft_toggle),
            ],
        "Selection":
            [
                ("Select all", "Control-a", select_all),
                ("Flip", ")", flip),
                ("Mirror", "(", mirror),
                ("Gamma", "g", gamma),
                ("Normalize", "n", normalize),
                ("Equalize", "e", equalize),
                ("Adaptive Equalize", "E", adaptive_equalize),
                ("Multiply", "m", multiply),
                ("Contrast", "c", contrast),
                ("Add", "a", add),
                ("Invert", "i", invert),
                ("Sigmoid", "I", sigmoid),
                ("Unsharp mask", "M", unsharp_mask),
                ("Blur", "B", blur),
                ("Highpass", "H", highpass),
                ("Clip light", "l", clip_high),
                ("Clip dark", "d", clip_low),
                ("Tres light", "L", tres_high),
                ("Tres dark", "D", tres_low),
                ("delete", "Delete", delete),
                ("fill", "Insert", fill),
            ],
        "View":
            [
                ("Histogram", "h", hist_toggle),
                ("Stats", "t", stats_toggle),
#                ("Zoom in", "KP_Add", app.zoom_in),
#                ("Zoom out", "KP_Subtract", app.zoom_out),
            ],
    }


def buttons_dict():
    return [
#        ("+", app.zoom_in),
#        ("-", app.zoom_out),
        ("Open", load),
        ("Save as", save_as),
        ("Undo", undo),
        ("Redo", redo),
        ("Histogram", hist_toggle),
        ("Statistics", stats_toggle),
        ("Crop", crop),
        ("Rotate", rotate_90),
        ("Normalize", normalize),
        ("Gamma", gamma),

    ]




#  ------------------------------------------
#  FILE
#  ------------------------------------------

def load(fp=None):
    logging.info("open")

    if not fp:
        logging.info("fpath input dialog")
        fp = filedialog.askopenfilename()
    if not fp:
        return

    app.img.load(fp)
    app.filelist = FileList(fp, extensions=CFG["image_extensions"])
    os.chdir(app.img.fpath.parent)
    app.history = nphistory.History(max_length=CFG["history_steps"]) # reset history
    app.history.original = app.img.arr.copy()
    app.history.add(app.img.arr, "load")
    app.title(app.img.fpath)
    app.reset()
    app.histwin.update()


def load_next():
    load(app.filelist.next)

def load_previous():
    load(app.filelist.previous)

def load_first():
    load(app.filelist.first)

def load_last():
    load(app.filelist.last)


def save():
    logging.info("save")
    app.img.save()


def save_as():
    logging.info("save as")
    app.img.save_as()
    app.title(app.img.fpath)


def save_as_png():
    logging.info("save as png")
    app.img.fpath = app.img.fpath.with_suffix(".png")
    app.img.save()


def toggle_original():

    #    app.img.reset()
    if not app.history.last():
        logging.info("nothing to toggle")
        return

    if not app.history.toggle_original:
        logging.info("show original")
        app.img.arr = app.history.original.copy()
    else:
        logging.info("show last")
        app.img.arr = app.history.last()['arr'].copy()

    app.history.toggle_original = not app.history.toggle_original
    app.update()

#  ------------------------------------------
#  HISTORY
#  ------------------------------------------


def undo():
    logging.info("undo")
    prev = app.history.undo()
    if prev:
        app.img.arr = prev['arr'].copy()
        app.update()
        app.histwin.update()
        app.statswin.update()


def redo():
    logging.info("redo")
    nex = app.history.redo()
    if nex:
        app.img.arr = nex['arr'].copy()
        app.update()
        app.histwin.update()
        app.statswin.update()

#  ------------------------------------------
#  IMAGE
#  ------------------------------------------


def edit_image(func):
    ''' decorator :
       apply changes, update gui and history '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(func.__name__)
        func(*args, **kwargs)
        logging.debug(f"edit_image {func.__name__} {args} {kwargs}")
        app.history.add(app.img.arr,  func.__name__, *args, **kwargs)
        app.update()
        app.selection.reset()
    return wrapper


@edit_image
def free_rotate():
    f = askfloat("Rotate angle (clockwise)", initialvalue=2.)
    app.img.free_rotate(-f)  # clockwise


@edit_image
def rotate_90():
    app.img.rotate()


@edit_image
def rotate_270():
    app.img.rotate(3)


@edit_image
def rotate_180():
    app.img.rotate(2)


@edit_image
def rgb2gray():
    app.img.rgb2gray()


@edit_image
def fft_toggle():
    app.img.fft_toggle()


#  ------------------------------------------
#  SELECTION
#  ------------------------------------------
def select_all():
    logging.info("select all")
    app.selection.reset()


def edit_selected(func):
    ''' decorator :
    load selection, ly changes, save to image, update gui and history '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        y = app.img.get_selection()
        try:
            y = func(y, *args, **kwargs)
            app.history.add(app.img.arr,  func.__name__)
        except Exception as e:
            logging.info(e) # ignore error (eg. dialog cancel)
            return
        app.img.set_selection(y)
        app.update()
        app.histwin.update()
        app.statswin.update()
#        app.history.add(app.img.arr,  func.__name__)
        logging.info("added to history")
    return wrapper


@edit_selected
def invert(y):
    return npfilters.invert(y)


@edit_selected
def mirror(y):
    return npfilters.mirror(y)


@edit_selected
def flip(y):
    return npfilters.flip(y)


@edit_selected
def contrast(y):
    f = askfloat("contrast", initialvalue=1.3)
    return npfilters.contrast(y, f)


@edit_selected
def multiply(y):
    f = askfloat("Multiply", initialvalue=1.3)
    return npfilters.multiply(y, f)


@edit_selected
def add(y):
    f = askfloat("Add", initialvalue=.2)
    if f is not None:
        return npfilters.add(y, f)


@edit_selected
def normalize(y):
    return npfilters.normalize(y)


@edit_selected
def adaptive_equalize(y):
    f = askfloat("adaptive_equalize clip limit", initialvalue=.02)
    return npfilters.adaptive_equalize(y, clip_limit=f)


@edit_selected
def equalize(y):
    return npfilters.equalize(y)


@edit_selected
def fill(y):
    f = askfloat("Fill with:", initialvalue=1)
    return npfilters.fill(y, f)


@edit_selected
def delete(y):
    if app.img.fft is not None:
        logging.info(app.selection.slice())
        app.img.fft[app.selection.slice()] = 0
    return npfilters.fill(y, 0)


@edit_selected
def unsharp_mask(y):
    r = askfloat("unsharp_mask - radius:", initialvalue=.5)
    a = askfloat("unsharp_mask - amount:", initialvalue=0.2)
    return npfilters.unsharp_mask(y, radius=r, amount=a)


@edit_selected
def blur(y):
    f = askfloat("gaussian blur radius:", initialvalue=1)
    return npfilters.blur(y, f)


@edit_selected
def highpass(y):
    f = askfloat("subtrack_background", initialvalue=20)
    return npfilters.highpass(y, f)


@edit_selected
def sigmoid(y):
    f = askfloat("Increase contrast with S-shape curve: (5-10)", initialvalue=5)
    return npfilters.sigmoid(y, gain=f)


@edit_selected
def gamma(y):
    f = askfloat("Set Gamma:", initialvalue=.8)
    return npfilters.gamma(y, f)


@edit_selected
def clip_high(y):
    f = askfloat("Cut high:", initialvalue=.9)
    return npfilters.clip_high(y, f)


@edit_selected
def clip_low(y):
    f = askfloat("Cut low:", initialvalue=.1)
    return npfilters.clip_low(y, f)


@edit_selected
def tres_high(y):
    f = askfloat("treshold high", initialvalue=.9)
    return npfilters.tres_high(y, f)


@edit_selected
def tres_low(y):
    f = askfloat("treshold low", initialvalue=.1)
    return npfilters.tres_low(y, f)


def crop():
    logging.info(f"{app.selection} crop")
    app.img.crop(*app.selection.geometry)
    app.update()
    app.history.add(app.img.arr, "crop")
    app.selection.reset()


def circular_mask():
    logging.info("zoom in")
    app.selection.make_cirk_mask()

#  ------------------------------------------
#  GUI FUNCTIONS
#  ------------------------------------------



def get_mouse():
    ''' get mouse position relative to canvas top left corner '''
    x = int(app.canvas.winfo_pointerx() - app.canvas.winfo_rootx())
    y = int(app.canvas.winfo_pointery() - app.canvas.winfo_rooty())
    return x, y


def hist_toggle():
    toggle_win(app.histwin)


def stats_toggle():
    toggle_win(app.statswin)


def keyPressed(event):
    ''' hotkeys '''

    for menu, items in commands_dict().items():
        for name, key, command in items:
            if event.keysym == key:
                command()


def toggle_win(win):

    if win.hidden:
        win.deiconify()
        win.hidden = False
        win.update()  # works only when not hidden

    else:
        win.withdraw()
        win.hidden = True

    app.focus_set()
    app.focus_force()


#  ------------------------------------------
#  MAIN WINDOW
#  ------------------------------------------


class App(tk.Toplevel):

    def __init__(self, master=None,  img_path=None, img_arr=None, fft=None):
        super().__init__(master)

        self.master = master
        self.geometry("900x810")
        self.img = npimage.npImage(img_path=img_path, img_arr=img_arr, fft=fft)
        self.filelist = FileList(img_path, extensions=CFG["image_extensions"])
        self.zoom_var = tk.StringVar()
        self.zoom = 1
        self.ofset = [0, 0]

        self.selection = Selection(master=self)
        self.history = nphistory.History(max_length=CFG["history_steps"])
        self.histwin = nphistwin.histWin(
            master=self, hide=CFG["hide_histogram"])
        self.statswin = npstatswin.statsWin(
            master=self, hide=CFG["hide_stats"])

        self.history.add(self.img.arr, "orig")
        self.history.original = self.img.arr.copy()

        self._gui_toolbar_init()

        self._gui_menu_init()
        self._gui_canvas_init()
        self._gui_bind_keys()

        self.reset()

    def _gui_menu_init(self):
        if CFG["hide_toolbar"]:
            return
        self.menubar = tk.Menu(self)
        tkmenu = {}
        for submenu, items in commands_dict().items():
            tkmenu[submenu] = tk.Menu(self.menubar, tearoff=0)
            for name, key, command in items:
                tkmenu[submenu].add_command(label=f"{name}   {key}",
                                                  command=command)
            self.menubar.add_cascade(label=submenu, menu=tkmenu[submenu])
            self.config(menu=self.menubar)



    def _gui_toolbar_init(self):

        backgroundColour = "white"
        buttonWidth = 6
        buttonHeight = 1

        self.toolbar = tk.Frame(self)
        tk.Label(self.toolbar, width=buttonWidth, text="gamma").pack(side="top")
        self.gamma_view_value = tk.DoubleVar(value=1.)

        self.gamma_view = tk.Entry(self.toolbar, width=buttonWidth, textvariable=self.gamma_view_value)
        self.gamma_view.pack(side="top")

        self.zoom_label = tk.Label(
            self.toolbar, width=buttonWidth, textvariable=self.zoom_var)
        self.zoom_label.pack(side="top")

        for i, b in enumerate(buttons_dict()):
            button = tk.Button(self.toolbar, text=b[0], font=('Arial Narrow', '10'),
                               background=backgroundColour, width=buttonWidth,
                               height=buttonHeight, command=b[1])
            button.pack(side="top")

        self.toolbar.pack(side="left")


    def _gui_bind_keys(self):

        self.protocol("WM_DELETE_WINDOW", self._quit)

        self.bind("<Key>", lambda event: keyPressed(event))
        self.bind("<Escape>", lambda event: select_all())
        self.bind("<MouseWheel>", self._mouse_wheel)  # windows
        self.bind("<Button-4>", self._mouse_wheel)  # linux
        self.bind("<Button-5>", self._mouse_wheel)  # linux
        self.bind("<Control-Button-1>", lambda event: self.selection.set_border(b="NW"))
        self.bind("<Control-Button-3>", lambda event: self.selection.set_border(b="SE"))
        self.bind("<Control-Left>", lambda event: self.selection.set_border(b="W"))
        self.bind("<Control-Right>", lambda event: self.selection.set_border(b="E"))
        self.bind("<Control-Up>", lambda event: self.selection.set_border(b="N"))
        self.bind("<Control-Down>", lambda event: self.selection.set_border(b="S"))

    def _gui_canvas_init(self):
        width = 800
        height = 800
        self.canvas = tk.Canvas(self, width=width,
                                height=height, background="gray")
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.zoom = max(1, min(self.img.width // 2 **
                               9, self.img.height // 2**9))
        self.ofset = [0, 0]  # position of image NW corner relative to canvas

#    @timeit
    def _make_image_view(self):

        logging.info(self.img.arr.shape)
        logging.info(self.zoom)

        view = self.img.arr[::self.zoom, ::self.zoom, ...]
        self.view_shape = view.shape[:2]

        view = self._apply_view_filters(view)
        view = Image.fromarray(img_as_ubyte(np.clip(view,0,1)))

        self.view = ImageTk.PhotoImage(view, master=self)

#    @timeit
    def _apply_view_filters(self, view):

        gamma
        g = self.gamma_view_value.get()
        if g != 1:
            view = npfilters.gamma(view, g)

        return view


    @timeit
    def draw(self):
        ''' draw new image '''
        self._make_image_view()
        logging.info(f"ofset {self.ofset}")
        self.image = self.canvas.create_image(self.ofset[0], self.ofset[1],
                                              anchor="nw", image=self.view)

    def reset(self):
        self.zoom = max(1, self.img.width//800,  self.img.height//800)
        self.ofset = [0, 0]
        # logging.info(f"initial zoom set: {self.zoom}")
        self.update()

    @timeit
    def update(self):
        ''' update image '''
        self.draw()
        self.title(self.img.properties())
#        self.histwin.update()
#        self.statswin.update()

    def _mouse_draw(self, event):
        ''' not implemented '''

    # def _mouse_select_left(self, event):
        # x, y = get_mouse()
        # if x < 0 or y < 0:
            # return
        # self.selection.set_topleft()

    # def _mouse_select_right(self, event):
        # x, y = get_mouse()
        # if x < 0 or y < 0:
            # return
        # self.selection.set_bottomright()

    def _mouse_wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get event coords
        y = self.canvas.canvasy(event.y)
        if event.num == 4 or event.delta == +120:
            self.zoom_in(x, y)
        if event.num == 5 or event.delta == -120:
            self.zoom_out(x, y)

    @property
    def ofset(self):
        return self.__dict__['ofset']

    @ofset.setter
    def ofset(self, coords):
        ofset = coords
        if hasattr(self, "canvas"):
            #            view_wid = self.img.width / self.zoom
            max_offset = [-self.img.width / self.zoom + self.canvas.winfo_width(),
                          -self.img.height / self.zoom + self.canvas.winfo_height()]

            # will whole canvas
            ofset = [max(max_offset[i], c) for i, c in enumerate(ofset)]
        ofset = [min(0, c) for c in ofset]  # only allow negative ofset
        self.__dict__['ofset'] = ofset

    # ensure zoom > 0
    @property
    def zoom(self):
        return self.__dict__['zoom']

    @zoom.setter
    def zoom(self, value):
        if value > 0:
            self.__dict__['zoom'] = int(value)
            self.zoom_var.set(f"{100/value:.0f} %")



    def zoom_out(self, x=0, y=0):
        logging.info("zoom out")
        if self.zoom < 50:
            old_zoom = app.zoom
            self.zoom += self.zoom_step()  #
            view_wid = app.img.width / app.zoom
            logging.info(f'view_wid {view_wid}')
            self.ofset = [c * self.zoom / old_zoom for c in self.ofset]

    #        app.ofset = [x, y]
            self.update()
            self.selection.reset()


    def zoom_in(self, x=0, y=0):
        ''' zoom in in allowed steps,
        put pixel with mouse pointer to canvas center'''
        logging.info("zoom in")
        if self.zoom > 1:
            # get canvas center
            zs = self.zoom_step()  #
            self.zoom -= zs
            ccy, ccx = [self.canvas.winfo_width() / 2,
                        self.canvas.winfo_height() / 2]
            magnif_change = 1 + zs / self.zoom
            # calculate ofset so that center of image is in center of canvas
            ofset = [ccx - x * magnif_change,
                     ccy - y * magnif_change]
            self.ofset = [ofset[0] + self.ofset[0],
                         ofset[1] + self.ofset[1]]
            logging.info(f"xy {x} {y} canvas c {ccx} {ccy} ofset {self.ofset} \
            mofset {self.ofset} zoom {self.zoom} \
            zoom step {self.zoom_step()} magnif_change {magnif_change}")

            self.update()
            self.selection.reset()


    def zoom_step(self):
        return int(self.zoom ** 1.5 / 10+1)

    def _quit(self):
        print("close window")
        # self.destroy()  # keep mainloop running
        self.quit()  # exit mainloop
        # sys.exit()

#  ------------------------------------------
#  Selection
#  ------------------------------------------


class Selection:

    def __init__(self, master):
        self.master = master
        self.geometry = [0, 0, 0, 0]
        self.rect = None
        self.mask = None
        self.cirk_mode = False

    def slice(self):
        ''' recalculate selection by zoom '''
        x0, y0, x1, y1 = [app.zoom * c for c in self.geometry]
        slice = np.s_[y0:y1, x0:x1, ...]
        app.img.slice = slice

        return slice

    def set_border(self, b="", *args, **kwargs):
        if "N" in b:
            self.geometry[1] = list(get_mouse())[1]
        if "E" in b:
            self.geometry[2] = list(get_mouse())[0]
        if "S" in b:
            self.geometry[3] = list(get_mouse())[1]
        if "W" in b:
            self.geometry[0] = list(get_mouse())[0]
        self.draw()

    def draw(self):
        self._validate_selection()
        app.canvas.delete(self.rect)
        self.rect = app.canvas.create_rectangle(self.geometry, outline='red')
        self.slice()
        if self.cirk_mode:
            self.make_cirk_mask()

    def _validate_selection(self):

        # avoid right corner being before left
        geomx = sorted(self.geometry[0::2])
        geomy = sorted(self.geometry[1::2])

        # keep selection in image
        x0, y0, x1, y1 = geomx[0], geomy[0], geomx[1], geomy[1]
        x0 = max(x0, 0)
        y0 = max(y0, 0)
        xmax, ymax = app.img.width * app.zoom, app.img.height * app.zoom
        x1 = min(x1, xmax)
        y1 = min(y1, ymax)
        self.geometry = [x0, y0, x1, y1]
        self.area = (x1 - x0) * (y1 - y0)
        print(f"selection area {self.area} pixels")

    def reset(self):
        self.select_all()
        self.draw()

    def select_all(self):
        self.geometry = [0, 0, app.img.width * app.zoom,
                         app.img.height * app.zoom]

    def make_cirk_mask(self):
        x0, y0, x1, y1 = self.geometry
        b, a = (x1+x0)/2, (y1+y0)/2
        nx, ny = app.img.arr.shape[:2]

        y, x = np.ogrid[-a:nx-a, -b:ny-b]
        radius = x0 - b
        mask = x*x + y*y <= radius*radius

        return mask

    def __str__(self):
        return f"selection geom: {self.geometry}"
#  ------------------------------------------
#  MAIN
#  ------------------------------------------


if __name__ == '__main__':

    logging.basicConfig(filename=LOG_FPATH, filemode='w',
        level=20,
        format='%(relativeCreated)d !%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')

    # get filename from command line argument or sample
    if len(sys.argv) > 1:
        Fp = Path(sys.argv[1].strip("'").strip('"'))
        assert Fp.is_file(), f"not a file {Fp}"
    else:
        Fp = Path(__file__).parent / 'sample.jpg'

    root = tk.Tk()
    root.title("Npyshop")
    root.withdraw()  # root win is hidden

    app = App(root, img_path=Fp)
    app.focus_set()

    logging.info(f"mainloop in {time.time()-time0}")

    root.mainloop()
    
    print("quit npyshop")

