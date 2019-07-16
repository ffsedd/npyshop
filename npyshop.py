#!/usr/bin/env python3
import os
from testing.timeit import timeit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from skimage import img_as_ubyte
from skimage import exposure  # histogram plotting, equalizing

from matplotlib import pyplot as plt
from npimage import npImage
from nphistory import History
import nputils
import npfilters
import tkinter as tk
import numpy as np
from pathlib import Path
import logging
import sys
import time
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
    editable FFT - new mainwin

    apply clipping only when necessary

'''


SETTINGS = {
    "hide_histogram": True,
    "hide_toolbar": True,
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
                ("Original", "q", original),

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
                ("Select left corner", "comma", select.set_left),
                ("Select right corner", "period", select.set_right),
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
        ("Undo", undo),
        ("Histogram", hist_toggle),
        ("Statistics", stats_toggle),
        ("Crop", crop),
        ("Rotate", rotate_90),
        ("Zoom in", zoom_in),
        ("Zoom out", zoom_out),
    ]


#  ------------------------------------------
#  FILE
#  ------------------------------------------

def load(fp=None):
    logging.info("open")
    img.load(fp)
    os.chdir(img.fpath.parent)
    history.original = img.arr.copy()
    history.add(img.arr,"load")
    mainwin.title(img.fpath)
    mainwin.reset()
    histwin.reset()


def save():
    logging.info("save")
    img.save()


def save_as():
    logging.info("save as")
    img.save_as()
    mainwin.title(img.fpath)


def save_as_png():
    logging.info("save as png")
    img.fpath = img.fpath.with_suffix(".png")
    img.save()


def original():

#    img.reset()
    if not history.last():
        logging.info("nothing to toggle")
        return

    if not history.toggle_original:
        logging.info("show original")
        img.arr = history.original.copy()
    else:
        logging.info("show last")
        img.arr = history.last()['arr'].copy()

    history.toggle_original = not history.toggle_original
    mainwin.update()

#  ------------------------------------------
#  HISTORY
#  ------------------------------------------


def undo():
    logging.info("undo")
    prev = history.undo()
    if prev:
        img.arr = prev['arr'].copy()
        mainwin.update()


def redo():
    logging.info("redo")
    nex = history.redo()
    if nex:
        img.arr = nex['arr'].copy()
        mainwin.update()

#  ------------------------------------------
#  IMAGE
#  ------------------------------------------


def edit_image(func):
    ''' decorator :
   apply changes, update gui and history '''
    def wrapper(*args, **kwargs):
        logging.info(func.__name__)
        func(*args, **kwargs)
        mainwin.update()
        history.add(img.arr,  func.__name__)
        select.reset()
    return wrapper


@edit_image
def free_rotate():
    f = tk.simpledialog.askfloat("Rotate", "Angle (float - clockwise)",
                                 initialvalue=2.)
    img.free_rotate(-f)  # clockwise


@edit_image
def rotate_90():
    img.rotate()


@edit_image
def rotate_270():
    img.rotate(3)


@edit_image
def rotate_180():
    img.rotate(2)


@edit_image
def rgb2gray():
    img.rgb2gray()


#  ------------------------------------------
#  SELECTION
#  ------------------------------------------
def select_all():
    logging.info("select all")
    select.reset()


def edit_selected(func):
    ''' decorator :
   load selection, apply changes, save to image, update gui and history '''
    def wrapper(*args, **kwargs):
        print("edit_selected: ",func.__name__)
        y = img.get_selection()
        y = func(y, *args, **kwargs)
        img.set_selection(y)
        mainwin.update()
        history.add(img.arr,  func.__name__)
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
    fft_arr = img.fft()
    fft_img = npImage(arr=fft_arr)
    fft_img.arr = nputils.normalize(fft_img.arr)
    mainWin(master=root,  curr_img=fft_img)


@edit_selected
def ifft():
    ifft_arr = img.ifft()
    ifft_img = npImage(arr=ifft_arr)
    ifft_img.arr = nputils.normalize(ifft_img.arr)
    mainWin(master=root,  curr_img=ifft_img)


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
    logging.info(f"{select} crop")
    img.crop(*select.geometry)
    mainwin.update()
    history.add(img.arr, "crop")
    select.reset()


def circular_mask():
    logging.info("zoom in")
    select.make_cirk_mask()


def zoom_out(x, y):
    logging.info("zoom out")
    if mainwin.zoom < 50:
        old_zoom = mainwin.zoom
        mainwin.zoom += zoom_step()  #
        view_wid = img.width / mainwin.zoom
        print('view_wid', view_wid)
        mainwin.ofset = [c * mainwin.zoom / old_zoom for c in mainwin.ofset]

#        mainwin.ofset = [x, y]
        mainwin.update()
        select.reset()


def zoom_in(x, y):
    ''' zoom in in allowed steps,
    put pixel with mouse pointer to canvas center'''
    logging.info("zoom in")
    if mainwin.zoom > 1:
        # get canvas center
        zs = zoom_step()  #
        mainwin.zoom -= zs
        ccy, ccx = [mainwin.canvas.winfo_width() / 2,
                    mainwin.canvas.winfo_height() / 2]
        magnif_change = 1 + zs / mainwin.zoom
        # calculate ofset so that center of image is in center of canvas
        ofset = [ccx - x * magnif_change,
                 ccy - y * magnif_change]
        mainwin.ofset = [ofset[0]+mainwin.ofset[0],
                         ofset[1]+mainwin.ofset[1]]
        logging.info(f"xy {x} {y} canvas c {ccx} {ccy} ofset {ofset} \
        mofset {mainwin.ofset} zoom {mainwin.zoom} \
        zoom step {zoom_step()} magnif_change {magnif_change}")

        mainwin.update()
        select.reset()


def zoom_step():
    return int(mainwin.zoom**1.5/10+1)


#  ------------------------------------------
#  GUI FUNCTIONS
#  ------------------------------------------


def get_mouse():
    ''' get mouse position relative to canvas top left corner '''
    x = int(mainwin.canvas.winfo_pointerx() - mainwin.canvas.winfo_rootx())
    y = int(mainwin.canvas.winfo_pointery() - mainwin.canvas.winfo_rooty())
    return x, y


def hist_toggle():
    toggle_win(histwin)


def stats_toggle():
    toggle_win(statswin)


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

    mainwin.focus_force()


def quit_app():

    logging.info("quit app")
    root.destroy()
    root.quit()
#    sys.exit()


#  ------------------------------------------
#  FFT
#  ------------------------------------------


class plotWin(tk.Toplevel):

    def __init__(self, master=None, plot=None, *a, **kw):
        super().__init__(master)
        self.title("Numpyshop-plot")
        self.master = master
        self.plot = plot
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.geometry("300x300")
        self.bind("<Control-s>", self._save)
        self.bind("<Key>", lambda event: keyPressed(event))
        self.draw(*a, **kw)

    def _save(self, event):
        #        self.fig.savefig(str(img.fpath) + "_plot.png")
        y = self.plot
        nputils.info(y)
#        y = nputils.normalize(self.plot)
        nputils.save_image(y, str(img.fpath) +
                           "_plot.png", bitdepth=img.bitdepth)
#    @timeit

    def draw(self, *a, **kw):
        self.fig = plt.figure(figsize=(5, 5))
        self.im = plt.imshow(self.plot, cmap='gray',
                             interpolation=None, *a, **kw)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


#  ------------------------------------------
#  HISTOGRAM
#  ------------------------------------------


class histWin(tk.Toplevel):
    
    @timeit
    def __init__(self, master=None, linewidth=1.0):
        super().__init__(master)
        self.title("Histogram")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", hist_toggle)
        self.geometry("300x300")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.linewidth = linewidth
        self.bins = SETTINGS["histogram_bins"]
        self.hidden = SETTINGS["hide_histogram"]
        self.draw()
        if self.hidden:
            self.withdraw()
            
    @timeit
    def draw(self):

        self.fig, self.ax_hist = plt.subplots()
#        self.ax_hist.set_xticks(np.linspace(0, 1, 11))
        self.ax_hist.set_xlim(0, 1)
        self.ax_hist.set_ylim(0, 10)
#        self.ax_hist.set_title("Histogram")

        # Display histogram
        self.ax_hist.spines['right'].set_visible(False)
        self.ax_hist.spines['top'].set_visible(False)
        self.ax_hist.spines['left'].set_visible(False)
        self.ax_hist.tick_params(left=False)
        # self.ax_hist.hist(img.arr.ravel(), bins=self.bins, range=(0, 1),
                          # density=True, histtype='step', color='black')

        # Display cumulative distribution
        self.ax_cdf = self.ax_hist.twinx()
        self.ax_cdf.spines['right'].set_visible(False)
        self.ax_cdf.spines['top'].set_visible(False)
        self.ax_cdf.spines['left'].set_visible(False)
        self.ax_cdf.tick_params(left=False)
        # img_cdf, bins = exposure.cumulative_distribution(img.arr, self.bins)
        # self.ax_cdf.plot(bins, img_cdf, 'r')
        self.ax_cdf.set_yticks([])

        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#        self.ax_hist.plot(self.x, 0*self.x, color=color)[0]

    def reset(self):
        self.update()

    @timeit
    def update(self):

        if self.hidden:
            return

        self.ax_hist.cla()
        self.ax_cdf.cla()
        self.ax_hist.hist(img.arr.ravel(), bins=self.bins, range=(0, 1),
                          density=True, histtype='step', color='black')
        img_cdf, bins = exposure.cumulative_distribution(img.arr, self.bins)
        self.ax_cdf.plot(bins, img_cdf, 'r')
        self.canvas.draw()

#  ------------------------------------------
#  STATISTICS
#  ------------------------------------------


class statsWin(tk.Toplevel):
    @timeit
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop-stats")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", stats_toggle)
        self.geometry("150x230")
        self.bind("<Key>", lambda event: keyPressed(event))

        self.hidden = SETTINGS["hide_stats"]
        self.frame = tk.Frame(self)
        if self.hidden:
            self.withdraw()
        
        # self.draw()

    # def draw(self):
        # self.frame = tk.Frame(self)
        # self._draw_table()

    @timeit
    def update(self):

        if self.hidden:
            return
        self.frame.grid_forget()
        self._draw_table()

#    @timeit
    def _draw_table(self):

        for r, k in enumerate(img.stats):  # loop stats dictionary
            bg = "#ffffff" if r % 2 else "#ddffee"  # alternating row colors
            # keys
            b1 = tk.Label(self.frame, text=k, font=(None, 9),
                          background=bg, width=9)
            b1.grid(row=r, column=1)

            # values
            b2 = tk.Label(self.frame, text=img.stats[k], font=(None, 9),
                          background=bg, width=9)
            b2.grid(row=r, column=2)

        self.frame.pack(side=tk.LEFT)


#  ------------------------------------------
#  MAIN WINDOW
#  ------------------------------------------


class mainWin(tk.Toplevel):

    def __init__(self, master=None,  curr_img=None):
        super().__init__(master)
        self.img = curr_img
        # print(curr_img)
        self.master = master
        self.protocol("WM_DELETE_WINDOW", quit_app)
        self.geometry("900x810")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.bind("<Escape>", lambda event: select_all())
        self.bind("<MouseWheel>", self.__wheel)  # windows
        self.bind("<Button-4>", self.__wheel)  # linux
        self.bind("<Button-5>", self.__wheel)  # linux
        self.bind("<Button-1>", self._on_mouse_left)
        self.bind("<Button-3>", self._on_mouse_right)
        self.zoom = 1
        print("zoom",  self.zoom)
        self.ofset = [0, 0]
        print("ofset ", self.ofset)
        if not SETTINGS["hide_toolbar"]:
            self.buttons_init()
        self.menu_init()
        self.canvas_init()
        self.reset()

    def __wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)  # get event coords
        y = self.canvas.canvasy(event.y)
        if event.num == 4 or event.delta == +120:
            zoom_in(x, y)
        if event.num == 5 or event.delta == -120:
            zoom_out(x, y)

    def buttons_init(self):

        backgroundColour = "white"
        buttonWidth = 6
        buttonHeight = 1
        self.toolbar = tk.Frame(self)

        for i, b in enumerate(buttons_dict()):
            button = tk.Button(self.toolbar, text=b[0],
                               background=backgroundColour, width=buttonWidth,
                               height=buttonHeight, command=b[1])
            button.grid(row=i, column=0)

        self.toolbar.pack(side=tk.LEFT)

    def menu_init(self):

        self.menubar = tk.Menu(self)
        tkmenu = {}
        for submenu, items in commands_dict().items():
            tkmenu[submenu] = tk.Menu(self.menubar, tearoff=0)
            for name, key, command in items:
                tkmenu[submenu].add_command(label=f"{name}   {key}",
                                                  command=command)
            self.menubar.add_cascade(label=submenu, menu=tkmenu[submenu])
            self.config(menu=self.menubar)

    def canvas_init(self):
        width = 800
        height = 800
        self.canvas = tk.Canvas(self, width=width,
                                height=height, background="gray")
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.zoom = max(1, min(img.width // 2**9, img.height // 2**9))
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
        self.zoom = max(1, img.width//800,  img.height//800)
        self.ofset = [0, 0]
        # print(f"initial zoom set: {self.zoom}")
        self.update()

    @timeit
    def update(self):
        ''' update image '''
        self.draw()
        self.title(img.properties())
        histwin.update()
        statswin.update()

    def on_draw(self, event):
        ''' track current selection coordinates '''
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()
        # print(self.xlim,self.ylim)

    def _on_mouse_left(self, event):
        select.set_left()

    def _on_mouse_right(self, event):
        select.set_right()

    @property
    def ofset(self):
        return self.__dict__['ofset']

    @ofset.setter
    def ofset(self, coords):
        ofset = coords
        if hasattr(self, "canvas"):
            view_wid = img.width / self.zoom
            # print('view_wid', view_wid)
            max_offset = [-img.width / self.zoom + self.canvas.winfo_width(),
                          -img.height / self.zoom + self.canvas.winfo_height()]
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
#  ------------------------------------------
#  Selection
#  ------------------------------------------


class Selection:
    
    @timeit
    def __init__(self, parent):
        self.parent = parent
        self.geometry = [0, 0, 0, 0]
        self.rect = None
        self.mask = None
        self.cirk_mode = False

    def slice(self):
        ''' recalculate selection by zoom '''
        x0, y0, x1, y1 = [mainwin.zoom * c for c in self.geometry]
        slice = np.s_[y0:y1, x0:x1, ...]
        img.slice = slice

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
        mainwin.canvas.delete(self.rect)
        self.rect = mainwin.canvas.create_rectangle(self.geometry)
#        print(self.rect)
        self.slice()
        if self.cirk_mode:
            self.make_cirk_mask()
        histwin.update()
    
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
        self.geometry = [0, 0, img.width * mainwin.zoom,
                         img.height * mainwin.zoom]


    def make_cirk_mask(self):
        x0, y0, x1, y1 = self.geometry
        b, a = (x1+x0)/2, (y1+y0)/2
        nx, ny = img.arr.shape[:2]

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

    if len(sys.argv) > 1:
        Fp = Path(sys.argv[1].strip("'").strip('"'))
        assert Fp.is_file(), f"not a file {Fp}"
    else:
        Fp = Path(__file__).parent / 'sample.jpg'

    root = tk.Tk()
    root.title("Numpyshop")
    root.withdraw()  # root win is hidden

    history = History(max_length=SETTINGS["history_steps"])

    # load image into numpy array
    img = npImage(Fp)
    history.add(img.arr, "orig")
    history.original = img.arr

    logging.info("image loaded")

    select = Selection(root)
    logging.info(select)

    histwin = histWin(root)

    statswin = statsWin(root)

    mainwin = mainWin(root, curr_img=img)
    mainwin.focus_set()

    logging.info(f"mainloop in {time.time()-time0}")

    root.mainloop()
