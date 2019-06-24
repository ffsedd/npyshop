#!/usr/bin/env python3
import sys
from pathlib import Path

import tkinter as tk
import numpy as np
from npimage import npImage

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)

from collections import deque


'''
RESOURCES:
image operations tutorial:
    https://homepages.inf.ed.ac.uk/rbf/HIPR2/wksheets.htm


'''

'''
BUGS:
error in histogram scaling


'''


SETTINGS = {
    "hide_toolbar": True,
    "hide_histogram": True,
    "hide_stats": True,
}


def commands(self):
    ''' can not be set as a global, contains undefined functions '''
    return {
            "File":
            [
                ("Open", "o", load),
                ("Save", "S", save),
                ("Save as", "s", save_as),
                ("Reset", "R", reset),
            ],
            "Edit":
            [
                ("Undo", "z", undo),
                ("Redo", "y", redo),
                ("Crop", "C", crop),
                ("Rotate", "r", rotate),
            ],
            "View":
            [
                ("Histogram", "h", hist_toggle),
                ("Stats", "t", stats_toggle),
                ("Toolbar", "b", toolbar_toggle),
            ],
            "Filter":
            [
                ("Gamma", "g", gamma),
                ("Normalize", "n", normalize),
                ("Multiply", "m", multiply),
                ("Add", "a", add),
                ("Contrast", "c", contrast),
                ("Invert", "i", invert),
            ],
            }


#  ------------------------------------------
#  FUNCTIONS
#  ------------------------------------------


def load():
    print("open")
    img.load()
    img.info()
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


def contrast():
    print("contrast")
    f = tk.simpledialog.askfloat("Contrast", "Enter float value")
    img.contrast(f)
    mainwin.update()


def multiply():
    print("multiply")
    f = tk.simpledialog.askfloat("Multiply", "Enter float value")
    img.multiply(f)
    mainwin.update()


def add():
    print("add")
    f = tk.simpledialog.askfloat("Add", "Enter float value")
    img.add(f)
    mainwin.update()


def normalize():
    print("normalize")
    img.normalize()
    mainwin.update()


def gamma():
    print("gamma")
    g = tk.simpledialog.askfloat("Set Gamma", "Enter float value")
    img.gamma(g)
    mainwin.update()


def crop():
    x0, x1 = sorted(mainwin.xlim)
    y0, y1 = sorted(mainwin.ylim)
    print(f"call crop: {x0} {x1} {y0} {y1}")
    img.crop(x0, x1, y0, y1)
    mainwin.update()


def hist_toggle():
    toggle_win(histwin)


def stats_toggle():
    toggle_win(statswin)


def toolbar_toggle():
    toggle_win(toolbar)

#  ------------------------------------------
#  GUI FUNCTIONS
#  ------------------------------------------


def keyPressed(event):
    ''' hotkeys '''

    for menu, items in commands(toolbar).items():
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

        buttons_cfg = [
                        ("Crop", crop),
                        ("Rotate", rotate),
                ]

        for i, b in enumerate(buttons_cfg):
            button = tk.Button(toolKitFrame, text=b[0],
                               background=backgroundColour, width=buttonWidth,
                               height=buttonHeight, command=b[1])
            button.grid(row=0, column=i)

        toolKitFrame.pack(side=tk.LEFT)

    def menuInit(self):

        menubar = tk.Menu(self)
        tkmenu = {}
        for submenu, items in commands(self).items():
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
        self.undo_queue = deque([], 10)
        self.redo_queue = deque([], 10)

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

        self.draw()

    def draw(self):
        ''' draw new image '''
        self.fig = plt.figure(figsize=(5, 5))
        self.cmap = "gray" if img.channels == 1 else "jet"
        self.im = plt.imshow(img.arr, cmap=self.cmap)
        self.ax = self.fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('draw_event', self.on_draw)

        history.add()

    def update(self, add_to_history=True):
        ''' update image '''
        if len(img.arr.ravel()) == 0:
            print("array is empty")
            return

        img.info()
        print(f"update w:{img.width}, h:{img.height}")

        self.im.set_data(img.arr)

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

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop-histogram")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", hist_toggle)
        self.geometry("300x300")
        self.bind("<Key>", lambda event: keyPressed(event))

        self.bins = 30

        self.hidden = SETTINGS["hide_histogram"]
        if self.hidden:
            self.withdraw()

        self.draw()

    def x(self):
        return np.linspace(0, 2 ** img.bitdepth, self.bins)

    def on_closing(self):
        self.withdraw()  # hide only

    def toggle(self):
        toggle_win(self)

    def draw(self):

        self.fig = plt.figure(figsize=(2, 4))
        self.axes = self.fig.add_subplot(111)
        self.ims = []

        for line in self.hist_lines():
            self.ims.append(plt.plot(self.x(), line[0], color=line[1])[0])
#        print("self.ims",self.ims)
#        self.axes = plt.axes()
        self.ax = plt.gca()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update(self):

        if self.hidden:
            return

        # calculate max of all lines (autoscale did not work, scale manually)
        hist_max = 0

        # loop and update all lines
        for im, line in zip(self.ims, self.hist_lines()):
            im.set_data(self.x(), line[0])
            hist_max = max(hist_max, line[0].max())

        self.axes.set_xlim(0, 2 ** img.bitdepth)
        self.axes.set_ylim(0, hist_max)

        self.canvas.draw()

    def plot_hist(self, y, color="black"):

        h = np.histogram(y, bins=self.bins, density=True)[0]  # normalized
        return h, color

    def hist_lines(self):
        ''' return list of histogram y values (1D) '''
        lines = []
        if img.channels == 1:  # gray
            y = img.arr
            lines.append(self.plot_hist(y, color="black"))
        else:    # RGB image
            for i, color in enumerate(('red', 'green', 'blue')):
                y = img.arr[:, :, i]
                lines.append(self.plot_hist(y, color=color))
        return lines

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

    def min(self):
        return img.arr.min()

    def max(self):
        return img.arr.min()

    def on_closing(self):
        self.withdraw()  # hide only

    def toggle(self):
        toggle_win(self)

    def draw(self):

        self.frame = tk.Frame(self)
        self._draw_chart()

    def update(self):

        if self.hidden:
            return
        self.frame.grid_forget()
        self._draw_chart()

    def _draw_chart(self):

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

    history = History()

    histwin = histWin(root)
    statswin = statsWin(root)

    toolbar = Toolbar(root)

    mainwin = mainWin(root)

    root.mainloop()
