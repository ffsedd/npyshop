#!/usr/bin/env python3
import time
time0 = time.time()
print("started")
import sys
from pathlib import Path
import numpy as np
import tkinter as tk
from npimage import npImage

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from skimage import img_as_float, img_as_ubyte, img_as_uint
    
from PIL import Image, ImageTk

from collections import deque
print("imports done")

'''
RESOURCES:
image operations:
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/wksheets.htm
    https://web.cs.wpi.edu/~emmanuel/courses/cs545/S14/slides/lecture02.pdf

'''

'''
BUGS:

self.set_cursor(cursors.SELECT_REGION)
  File "/home/m/.local/lib/python3.6/site-packages/matplotlib/backends/_backend_tk.py", line 613, in set_cursor
    window = self.canvas.manager.window
AttributeError: 'FigureCanvasTkAgg' object has no attribute 'manager'

sometimes incorrect crop region

large images:
matplotlib imshow very slow
running out of memory
'''


SETTINGS = {
    "hide_toolbar": True,
    "hide_histogram": False,
    "hide_stats": True,
    "history_steps": 5,
}


def commands_dict():
    ''' can not be set as a global, contains undefined functions '''
    return {
            "File":
            [
                ("Open", "o", load),
                ("Save", "S", save),
                ("Save as", "s", save_as),
                ("Reset", "Q", reset),
            ],
            "Edit":
            [
                ("Undo", "z", undo),
                ("Redo", "y", redo),
                ("Crop", "C", crop),
                ("Rotate_90", "r", rotate),
                ("Rotate_270", "R", rotate_270),
                ("Rotate_180", "u", rotate_180),
            ],
            "Filter":
            [
                ("Gamma", "g", gamma),
                ("Normalize", "n", normalize),
                ("Multiply", "m", multiply),
                ("Add", "a", add),
                ("Invert", "i", invert),
                ("Sigma", "G", sigma),
                ("Clip light", "l", clip_high),
                ("Clip dark", "d", clip_low),
                ("Tres light", "L", tres_high),
                ("Tres dark", "D", tres_low),
            ],
            "View":
            [
                ("Histogram", "h", hist_toggle),
                ("Stats", "t", stats_toggle),
                ("Toolbar", "b", toolbar_toggle),
            ],
            }


def buttons_dict():
    return [
            ("Open", load),
            ("Undo", undo),
            ("Histogram", hist_toggle),
            ("Statistics", stats_toggle),
            ("Crop", crop),
            ("Rotate", rotate),
            ]


#  ------------------------------------------
#  FUNCTIONS
#  ------------------------------------------
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__qualname__, (te - ts) * 1000))
        return result
    return timed


def load():
    print("open")
    img.load()
#    img.info()
    mainwin.update()
    histwin.update()


def save():
    print("save")
    img.save()


def save_as():
    print("save as")
    img.save_as()


def reset():
    print("reset")
    img.reset()
    mainwin.update()


def undo():
    print("undo")
    history.undo()
    mainwin.update(add_to_history=False)


def redo():
    print("redo")
    history.redo()
    mainwin.update(add_to_history=False)


def rotate():
    print("rotate")
    img.rotate()
    mainwin.update()


def rotate_270():
    print("rotate left")
    img.rotate(3)
    mainwin.update()


def rotate_180():
    print("rotate 180")
    img.rotate(2)
    mainwin.update()


def invert():
    print("invert")
    img.invert()
    mainwin.update()


def mirror():
    print("mirror")
    img.mirror()
    mainwin.update()


def flip():
    print("flip")
    img.flip()
    mainwin.update()


def multiply():
    print("multiply")
    f = tk.simpledialog.askfloat("Multiply", "Value to multiply with (float)",
                                 initialvalue=1.3)
    img.multiply(f)
    mainwin.update()


def add():
    print("add")
    f = tk.simpledialog.askfloat("Add", "Enter value to add (float)")
    img.add(f)
    mainwin.update()


def normalize():
    print("normalize")
    img.normalize()
    mainwin.update()


def sigma():
    print("sigma")
    g = tk.simpledialog.askfloat("Set Sigma", "Enter sigma (float)",
                                 initialvalue=2)
    img.sigma(g)
    mainwin.update()


def gamma():
    print("gamma")
    g = tk.simpledialog.askfloat("Set Gamma", "Enter gamma (float)",
                                 initialvalue=.8)
    img.gamma(g)
    mainwin.update()


def clip_high():
    print("clip_high")
    f = tk.simpledialog.askfloat("Cut high", "Enter high treshold (float)",
                                 initialvalue=.9)
    img.clip_high(f)
    mainwin.update()


def clip_low():
    print("clip_low")
    f = tk.simpledialog.askfloat("Cut low", "Enter low treshold (float)",
                                 initialvalue=.1)
    img.clip_low(f)
    mainwin.update()


def tres_high():
    print("tres_high")
    f = tk.simpledialog.askfloat("tres high", "Enter high treshold (float)",
                                 initialvalue=.9)
    img.tres_high(f)
    mainwin.update()


def tres_low():
    print("tres_low")
    f = tk.simpledialog.askfloat("tres low", "Enter low treshold (float)",
                                 initialvalue=.1)
    img.tres_low(f)
    mainwin.update()


def crop():
    x0, x1 = sorted(mainwin.xlim)
    y0, y1 = sorted(mainwin.ylim)
    print(f"call crop: {x0} {x1} {y0} {y1}")
    img.crop(x0, x1, y0, y1)
    mainwin.update()


#  ------------------------------------------
#  GUI FUNCTIONS
#  ------------------------------------------


def hist_toggle():
    toggle_win(histwin)


def stats_toggle():
    toggle_win(statswin)


def toolbar_toggle():
    toggle_win(toolbar)


def keyPressed(event):
    ''' hotkeys '''

    for menu, items in commands_dict().items():
        for name, key, command in items:
            if event.keysym == key:
                command()


def toggle_win(win):
    ''' shared method for floating windows,
    I tried to make new class and subclass it,
    but keybindings did not work then '''

    if win.hidden:
        win.deiconify()
        win.hidden = False
        win.update()  # works only when not hidden
    else:
        win.withdraw()
        win.hidden = True

    mainwin.focus_force()


def quit_app():

    print("quit app")
    root.destroy()


#  ------------------------------------------
#  TOOLBAR
#  ------------------------------------------


class Toolbar(tk.Toplevel):

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop-toolbar")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", toolbar_toggle)
        self.geometry("600x30")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.ButtonsInit()
        self.menuInit()

        self.hidden = SETTINGS["hide_toolbar"]

        if self.hidden:
            self.withdraw()

    def ButtonsInit(self):

        backgroundColour = "white"
        buttonWidth = 6
        buttonHeight = 1
        toolKitFrame = tk.Frame(self)

        for i, b in enumerate(buttons_dict()):
            button = tk.Button(toolKitFrame, text=b[0],
                               background=backgroundColour, width=buttonWidth,
                               height=buttonHeight, command=b[1])
            button.grid(row=0, column=i)

        toolKitFrame.pack(side=tk.LEFT)

    def menuInit(self):

        menubar = tk.Menu(self)
        tkmenu = {}
        for submenu, items in commands_dict().items():
            tkmenu[submenu] = tk.Menu(menubar, tearoff=0)
            for name, key, command in items:
                tkmenu[submenu].add_command(label=f"{name}   {key}",
                                                  command=command)
            menubar.add_cascade(label=submenu, menu=tkmenu[submenu])
            self.config(menu=menubar)

#  ------------------------------------------
#  HISTORY
#  ------------------------------------------


class History():

    def __init__(self):
        self.undo_queue = deque([], SETTINGS["history_steps"])
        self.redo_queue = deque([], SETTINGS["history_steps"])

    def add(self):
        self.undo_queue.append(img.arr.copy())
        self.redo_queue.clear()  # discard redo (new version)
        print(f"added to history, len:{len(self.undo_queue)}")

    def undo(self):
        if len(self.undo_queue) > 1:
            lastImage = self.undo_queue.pop()
            self.redo_queue.append(lastImage)
            img.arr = self.undo_queue[-1]

        print(f"undo queue len: {len(self.undo_queue)}")

    def redo(self):
        if len(self.redo_queue) > 0:
            img.arr = self.redo_queue.pop()
            self.undo_queue.append(img.arr.copy())

        print(f"redo queue len: {len(self.redo_queue)}")

        mainwin.update(add_to_history=False)

    def reset(self):
        img.arr = img.original.copy()
        mainwin.update()

#  ------------------------------------------
#  MAIN WINDOW
#  ------------------------------------------


class mainWin(tk.Toplevel):

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", quit_app)
        self.geometry("800x800")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.fig = None
        self.zoom = 8

        self.draw()

    @timeit
    def draw(self):
        ''' draw new image '''
        self.fig = plt.figure(figsize=(5, 5))
        self.cmap = "gray" if img.channels == 1 else "jet"
        self.view = img.arr[::self.zoom,::self.zoom,...]
        self.im = plt.imshow(self.view, cmap=self.cmap, interpolation=None)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('draw_event', self.on_draw)

        history.add()
        
    # @timeit
    # def draw(self):
        # ''' draw using pillow '''
        # self.zoom = 8 
        # self.imgtk =  ImageTk.PhotoImage(image=Image.fromarray(img_as_ubyte(img.arr[::self.zoom,::self.zoom,...])))

        # self.canvas = tk.Canvas(self,width=800,height=800)
        # self.canvas.pack()
        # self.canvas.create_image(0,0, anchor="nw", image=self.imgtk)
        
        # history.add()

    @timeit
    def update(self, add_to_history=True):
        ''' update image '''
        if len(img.arr.ravel()) == 0:
            print("array is empty")
            return

#        img.info()
        print(f"update w:{img.width}, h:{img.height}")

        self.im.set_data(img.view)

        # resize graph after crop, rotate
        self.im.set_extent((0, img.width, 0, img.height))
        self.ax.set_xlim(0, img.width)
        self.ax.set_ylim(0, img.height)

        self.canvas.draw()

        histwin.update()

        statswin.update()

        if add_to_history:
            history.add()

    def on_draw(self, event):
        ''' track current selection coordinates '''
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()
        # print(self.xlim,self.ylim)

#  ------------------------------------------
#  HISTOGRAM
#  ------------------------------------------


class histWin(tk.Toplevel):

    def __init__(self, master=None, linewidth=1.0):
        super().__init__(master)
        self.title("Numpyshop-histogram")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", hist_toggle)
        self.geometry("300x300")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.linewidth = linewidth
        self.bins = 30

        self.hidden = SETTINGS["hide_histogram"]
        if self.hidden:
            self.withdraw()

        self.draw()

    @timeit
    def draw(self):
        self.fig = plt.figure(figsize=(2, 4))
        self.axes = self.fig.add_subplot(111)

        plt.xticks(np.linspace(0, 1, 11), rotation='vertical')
#        self.ax.set_xtics(np.linspace(0, 1, 10))
#        self.fig = plt.figure(figsize=(2, 4))
#        self.axes = self.fig.add_subplot(111)
        self.ims = []

        x, hist_data = img.histogram_data()

        for color, y in hist_data.items():
            self.ims.append(plt.plot(x, y, color=color,
                                     linewidth=self.linewidth)[0])

        self.ax = plt.gca()
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(left=False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    @timeit
    def update(self):

        if self.hidden:
            return

        x, hist_data = img.histogram_data()
        # calculate max of all lines (autoscale did not work, scale manually)
        hist_max = 0
        # loop and update all lines
        for im, color in zip(self.ims, hist_data):
            y = hist_data[color]
            im.set_data(x, y)
            hist_max = max(hist_max, y.max())

        self.ax.set_ylim(0, hist_max)

        self.canvas.draw()


#  ------------------------------------------
#  STATISTICS
#  ------------------------------------------


class statsWin(tk.Toplevel):

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop-stats")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", stats_toggle)
        self.geometry("150x230")
        self.bind("<Key>", lambda event: keyPressed(event))

        self.hidden = SETTINGS["hide_stats"]
        if self.hidden:
            self.withdraw()

        self.draw()

    def draw(self):
        self.frame = tk.Frame(self)
        self._draw_table()

    @timeit
    def update(self):

        if self.hidden:
            return
        self.frame.grid_forget()
        self._draw_table()

    @timeit
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
#  MAIN
#  ------------------------------------------


if __name__ == '__main__':

    if len(sys.argv) > 1:
        Fp = Path(sys.argv[1])
        assert Fp.is_file(), f"not a file {Fp}"
    else:
        Fp = Path(__file__).parent / 'sample.tif'

    root = tk.Tk()
    root.title("Numpyshop")
    root.withdraw()  # root win is hidden

    # load image into numpy array
    img = npImage(Fp)
    print("image loaded")
    history = History()

    histwin = histWin(root)

    statswin = statsWin(root)

    toolbar = Toolbar(root)
    print("show mainwin")
    mainwin = mainWin(root)
    print(f"mainloop in {time.time()-time0}")

    root.mainloop()
