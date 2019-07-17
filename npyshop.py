#!/usr/bin/env python3
import logging
import sys
import time
import os

import tkinter as tk
import numpy as np
from pathlib import Path

from PIL import Image, ImageTk

from skimage import img_as_ubyte
#from skimage import exposure  # histogram plotting, equalizing

import npimage
import nphistory
import nputils
import npfilters
import nphistwin
import npstatswin
from testing.timeit import timeit

time0 = time.time()
print("imports done")

'''
RESOURCES:
image operations:
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/wksheets.htm
    https://web.cs.wpi.edu/~emmanuel/courses/cs545/S14/slides/lecture02.pdf


BUGS:

    large images and history on:
    running out of memory

TODO:
    circular selection?
    view area - crop before showing?
    editable FFT - new app

    apply clipping only when necessary

'''


CFG = {
    "hide_histogram": True,
    "hide_toolbar": False,
    "hide_stats": True,
    "histogram_bins": 256,
    "history_steps": 10,     # memory !!!

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
            ],
        "Selection":
            [
                ("Select left corner", "comma", selection_set_left),
                ("Select right corner", "period", selection_set_right),
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
                ("Sigma", "I", sigma),
                ("Unsharp mask", "M", unsharp_mask),
                ("Blur", "B", blur),
                ("Highpass", "H", highpass),
                ("Clip light", "l", clip_high),
                ("Clip dark", "d", clip_low),
                ("Tres light", "L", tres_high),
                ("Tres dark", "D", tres_low),
                ("FFT", "F", fft),
                ("iFFT", "Control-F", ifft),
                ("delete", "Delete", delete),
                ("fill", "Insert", fill),
            ],
        "View":
            [
                ("Histogram", "h", hist_toggle),
                ("Stats", "t", stats_toggle),
                ("Zoom in", "KP_Add", zoom_in),
                ("Zoom out", "KP_Subtract", zoom_out),
            ],
    }


def buttons_dict():
    return [
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
        ("Zoom in", zoom_in),
        ("Zoom out", zoom_out),
    ]


#  ------------------------------------------
#  FILE
#  ------------------------------------------

def load(fp=None):
    logging.info("open")
    app.img.load(fp)
    os.chdir(app.img.fpath.parent)
    app.history.original = app.img.arr.copy()
    app.history.add(app.img.arr,"load")
    app.title(app.img.fpath)
    app.reset()
    app.histwin.update()


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


def redo():
    logging.info("redo")
    nex = app.history.redo()
    if nex:
        app.img.arr = nex['arr'].copy()
        app.update()

#  ------------------------------------------
#  IMAGE
#  ------------------------------------------


def edit_image(func):
    ''' decorator :
   ly changes, update gui and history '''
    def wrapper(*args, **kwargs):
        logging.info(func.__name__)
        func(*args, **kwargs)
        app.update()
        app.history.add(app.img.arr,  func.__name__)
        app.selection.reset()
    return wrapper


@edit_image
def free_rotate():
    f = tk.simpledialog.askfloat("Rotate", "Angle (float - clockwise)",
                                 initialvalue=2.)
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


#  ------------------------------------------
#  SELECTION
#  ------------------------------------------
def select_all():
    logging.info("select all")
    app.selection.reset()


def edit_selected(func):
    ''' decorator :
   load selection, ly changes, save to image, update gui and history '''
    def wrapper(*args, **kwargs):
        print("edit_selected: ",func.__name__)
        y = app.img.get_selection()
        y = func(y, *args, **kwargs)
        app.img.set_selection(y)
        app.update()
        app.histwin.update()
        app.statswin.update()
        app.history.add(app.img.arr,  func.__name__)
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
    f = tk.simpledialog.askfloat("contrast", "Value to multiply with (float)",
                                 initialvalue=1.3)
    return npfilters.contrast(y, f)


@edit_selected
def multiply(y):
    f = tk.simpledialog.askfloat("Multiply", "Value to multiply with (float)",
                                 initialvalue=1.3)
    return npfilters.multiply(y, f)


@edit_selected
def add(y):
    f = tk.simpledialog.askfloat("Add", "Enter value to add (float)",
                                 initialvalue=.2)
    return npfilters.add(y, f)


@edit_selected
def normalize(y):
    return npfilters.normalize(y)


@edit_selected
def adaptive_equalize(y):
    f = tk.simpledialog.askfloat("adaptive_equalize", "clip limit (float)",
                                 initialvalue=.02)
    return npfilters.adaptive_equalize(y, clip_limit=f)


@edit_selected
def equalize(y):
    return npfilters.equalize(y)


@edit_selected
def fill(y):
    f = tk.simpledialog.askfloat("Fill", "Value to fill (float)",
                                 initialvalue=0)
    return npfilters.fill(y, f)


@edit_selected
def delete(y):
    return npfilters.fill(y, 1)


@edit_selected
def fft():
    fft_arr = app.img.fft()
    fft_img = npimage.npImage(arr=fft_arr)
    fft_img.arr = nputils.normalize(fft_img.arr)
    app(master=root,  img_arr=fft_img)


@edit_selected
def ifft():
    ifft_arr = app.img.ifft()
    ifft_img = npimage.npImage(arr=ifft_arr)
    ifft_img.arr = nputils.normalize(ifft_img.arr)
    app(master=root,  img_arr=ifft_img)


@edit_selected
def unsharp_mask(y):
    r = tk.simpledialog.askfloat("unsharp_mask", "Enter radius (float)",
                                 initialvalue=.5)
    a = tk.simpledialog.askfloat("unsharp_mask", "Enter amount (float)",
                                 initialvalue=0.2)
    return npfilters.unsharp_mask(y, radius=r, amount=a)


@edit_selected
def blur(y):
    f = tk.simpledialog.askfloat("blur", "Enter radius (float)",
                                 initialvalue=1)
    return npfilters.blur(y, f)


@edit_selected
def highpass(y):
    f = tk.simpledialog.askfloat("subtrack_background", "Enter sigma (float)",
                                 initialvalue=20)
    return npfilters.highpass(y, f)


@edit_selected
def sigma(y):
    f = tk.simpledialog.askfloat("Set Sigma", "Enter sigma (float)",
                                 initialvalue=3)
    return npfilters.sigma(y, f)


@edit_selected
def gamma(y):
    f = tk.simpledialog.askfloat("Set Gamma", "Enter gamma (float)",
                                 initialvalue=.8)
    return npfilters.gamma(y, f)


@edit_selected
def clip_high(y):
    f = tk.simpledialog.askfloat("Cut high", "Enter high treshold (float)",
                                 initialvalue=.9)
    return npfilters.clip_high(y, f)


@edit_selected
def clip_low(y):
    f = tk.simpledialog.askfloat("Cut low", "Enter low treshold (float)",
                                 initialvalue=.1)
    return npfilters.clip_low(y, f)


@edit_selected
def tres_high(y):
    f = tk.simpledialog.askfloat("tres high", "Enter high treshold (float)",
                                 initialvalue=.9)
    return npfilters.tres_high(y, f)


@edit_selected
def tres_low(y):
    f = tk.simpledialog.askfloat("tres low", "Enter low treshold (float)",
                                 initialvalue=.1)
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


def zoom_out(x, y):
    logging.info("zoom out")
    if app.zoom < 50:
        old_zoom = app.zoom
        app.zoom += zoom_step()  #
        view_wid = app.img.width / app.zoom
        print('view_wid', view_wid)
        app.ofset = [c * app.zoom / old_zoom for c in app.ofset]

#        app.ofset = [x, y]
        app.update()
        app.selection.reset()


def zoom_in(x, y):
    ''' zoom in in allowed steps,
    put pixel with mouse pointer to canvas center'''
    logging.info("zoom in")
    if app.zoom > 1:
        # get canvas center
        zs = zoom_step()  #
        app.zoom -= zs
        ccy, ccx = [app.canvas.winfo_width() / 2,
                    app.canvas.winfo_height() / 2]
        magnif_change = 1 + zs / app.zoom
        # calculate ofset so that center of image is in center of canvas
        ofset = [ccx - x * magnif_change,
                 ccy - y * magnif_change]
        app.ofset = [ofset[0]+app.ofset[0],
                         ofset[1]+app.ofset[1]]
        logging.info(f"xy {x} {y} canvas c {ccx} {ccy} ofset {ofset} \
        mofset {app.ofset} zoom {app.zoom} \
        zoom step {zoom_step()} magnif_change {magnif_change}")

        app.update()
        app.selection.reset()


def zoom_step():
    return int(app.zoom**1.5/10+1)


def selection_set_left():
    app.selection.set_left()


def selection_set_right():
    app.selection.set_right()


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

    def __init__(self, master=None,  img_path=None, img_arr=None):
        super().__init__(master)

        self.master = master
        self.geometry("900x810")

        self.img = npimage.npImage(img_path=img_path, img_arr=img_arr)
        self.zoom = 1
        self.ofset = [0, 0]

        self.selection = Selection(master=self)
        self.history = nphistory.History(max_length=CFG["history_steps"])
        self.histwin = nphistwin.histWin(master=self, hide=CFG["hide_histogram"])
        self.statswin = npstatswin.statsWin(master=self, hide=CFG["hide_stats"])

        self.history.add(self.img.arr, "orig")
        self.history.original = self.img.arr.copy()

        self._gui_buttons_init()
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

    def _gui_buttons_init(self):

        backgroundColour = "white"
        buttonWidth = 6
        buttonHeight = 1
        self.toolbar = tk.Frame(self)

        for i, b in enumerate(buttons_dict()):
            button = tk.Button(self.toolbar, text=b[0], font=('Arial Narrow', '10'),
                               background=backgroundColour, width=buttonWidth,
                               height=buttonHeight, command=b[1])
            button.grid(row=i, column=0)

        self.zoom_entry = tk.Entry(self.toolbar, width=buttonWidth, textvariable = self.zoom)
        self.zoom_entry.grid(row=i+1, column=0)

        self.toolbar.pack(side=tk.LEFT)

    def _gui_bind_keys(self):

        self.protocol("WM_DELETE_WINDOW", self._quit)

        self.bind("<Key>", lambda event: keyPressed(event))
        self.bind("<Escape>", lambda event: select_all())
        self.bind("<MouseWheel>", self._mouse_wheel)  # windows
        self.bind("<Button-4>", self._mouse_wheel)  # linux
        self.bind("<Button-5>", self._mouse_wheel)  # linux
        self.bind("<Button-1>", self._mouse_left)
        self.bind("<Button-3>", self._mouse_right)

    def _gui_canvas_init(self):
        width = 800
        height = 800
        self.canvas = tk.Canvas(self, width=width,
                                height=height, background="gray")
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.zoom = max(1, min(self.img.width // 2**9, self.img.height // 2**9))
        self.ofset = [0, 0]  # position of image NW corner relative to canvas

    @timeit
    def make_image_view(self):

        logging.info(self.img.arr.shape)
        logging.info(self.zoom)

        view = self.img.arr[::self.zoom, ::self.zoom, ...]
        self.view_shape = view.shape[:2]

        view = img_as_ubyte(view)
        view = Image.fromarray(view)
        self.view = ImageTk.PhotoImage(view, master=self)

    @timeit
    def draw(self):
        ''' draw new image '''
        self.make_image_view()
        print("ofset ", self.ofset)
        self.image = self.canvas.create_image(self.ofset[0], self.ofset[1],
                                              anchor="nw", image=self.view)

    def reset(self):
        self.zoom = max(1, self.img.width//800,  self.img.height//800)
        self.ofset = [0, 0]
        # print(f"initial zoom set: {self.zoom}")
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


    def _mouse_left(self, event):
        x,y = get_mouse()
        if x < 0 or y < 0:
            return
        self.selection.set_left()

    def _mouse_right(self, event):
        x,y = get_mouse()
        if x < 0 or y < 0:
            return
        self.selection.set_right()

    def _mouse_wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get event coords
        y = self.canvas.canvasy(event.y)
        if event.num == 4 or event.delta == +120:
            zoom_in(x, y)
        if event.num == 5 or event.delta == -120:
            zoom_out(x, y)

    @property
    def ofset(self):
        return self.__dict__['ofset']

    @ofset.setter
    def ofset(self, coords):
        ofset = coords
        if hasattr(self, "canvas"):
#            view_wid = self.img.width / self.zoom
            # print('view_wid', view_wid)
            max_offset = [-self.img.width / self.zoom + self.canvas.winfo_width(),
                          -self.img.height / self.zoom + self.canvas.winfo_height()]
            # print('max_offset', max_offset)
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


    def _quit(self):

        logging.info("quit")
        self.destroy()

#  ------------------------------------------
#  Selection
#  ------------------------------------------


class Selection:

    @timeit
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

    def set_left(self):
        self.geometry[:2] = list(get_mouse())
        self.draw()

    def set_right(self):
        self.geometry[2:] = list(get_mouse())
        self.draw()

    @timeit
    def draw(self):
        self._valid_selection()
#        print(self.geometry)
        app.canvas.delete(self.rect)
        self.rect = app.canvas.create_rectangle(self.geometry)
#        print(self.rect)
        self.slice()
        if self.cirk_mode:
            self.make_cirk_mask()
#        app.histwin.update()

    @timeit
    def _valid_selection(self):
        ''' avoid right corner being before left '''
#        if len(self.geometry) == 4:
        geomx = sorted(self.geometry[0::2])
        geomy = sorted(self.geometry[1::2])
        self.geometry = [geomx[0], geomy[0], geomx[1], geomy[1]]

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
        # print(y, x)
        radius = x0 - b
        # print(radius)
        mask = x*x + y*y <= radius*radius

        return mask

    def __str__(self):
        return f"selection geom: {self.geometry}"
#  ------------------------------------------
#  MAIN
#  ------------------------------------------


if __name__ == '__main__':

    logging.basicConfig(level=10, format='%(relativeCreated)d !%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')

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
