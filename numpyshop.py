#!/usr/bin/env python3
from testing.timeit import timeit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
from PIL import Image, ImageTk
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
from npimage import npImage
import tkinter as tk
import numpy as np
from pathlib import Path
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

color histogram weird

TODO:
    mouse zoom center on mouse pointer



'''


SETTINGS = {
    "hide_histogram": True,
    "hide_toolbar": True,
    "hide_stats": True,
    "histogram_bins": 32,
    "history_steps": 3,     # memory !!!

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
                ("Select left corner", "comma", select.set_left),
                ("Select right corner", "period", select.set_right),
                ("Rotate_90", "r", rotate),
                ("Rotate_270", "R", rotate_270),
                ("Rotate_180", "u", rotate_180),
                ("Free Rotate", "f", free_rotate),
            ],
        "Filter":
            [
                ("Gamma", "g", gamma),
                ("Normalize (BW only)", "n", normalize),
                ("Multiply", "m", multiply),
                ("Add", "a", add),
                ("Invert", "i", invert),
                ("Sigma", "I", sigma),
                ("Highpass", "H", highpass),
                ("Clip light (BW only)", "l", clip_high),
                ("Clip dark (BW only)", "d", clip_low),
                ("Tres light (BW only)", "L", tres_high),
                ("Tres dark (BW only)", "D", tres_low),
                ("FFT", "F", fft),
                ("rgb2gray", "b", rgb2gray),
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
        ("Rotate", rotate),
        ("Zoom in", zoom_in),
        ("Zoom out", zoom_out),
    ]


#  ------------------------------------------
#  FUNCTIONS
#  ------------------------------------------

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
    mainwin.update()


def redo():
    print("redo")
    history.redo()
    mainwin.update()


def free_rotate():
    print("free rotate")
    f = tk.simpledialog.askfloat("Rotate", "Angle (float)",
                                 initialvalue=2.)
    img.free_rotate(f)
    mainwin.update()
    history.add()
    select.reset()


def rotate():
    print("rotate")
    img.rotate()
    mainwin.update()
    history.add()
    select.reset()


def rotate_270():
    print("rotate left")
    img.rotate(3)
    mainwin.update()
    history.add()
    select.reset()


def rotate_180():
    print("rotate 180")
    img.rotate(2)
    mainwin.update()
    history.add()
    select.reset()


def invert():
    print("invert")
    img.invert()
    mainwin.update()
    history.add()


def mirror():
    print("mirror")
    img.mirror()
    mainwin.update()
    history.add()


def flip():
    print("flip")
    img.flip()
    mainwin.update()
    history.add()


def multiply():
    print("multiply")
    f = tk.simpledialog.askfloat("Multiply", "Value to multiply with (float)",
                                 initialvalue=1.3)
    img.multiply(f)
    mainwin.update()
    history.add()


def add():
    print("add")
    f = tk.simpledialog.askfloat("Add", "Enter value to add (float)",
                                 initialvalue=.2)
    img.add(f)
    mainwin.update()
    history.add()


def normalize():
    print("normalize")
    img.normalize()
    mainwin.update()
    history.add()


def rgb2gray():
    print("rgb2gray")
    img.rgb2gray()
    mainwin.update()
    history.add()


def fft():
    from matplotlib.colors import LogNorm
    print("fft")
    fftimage = img.fft()
    plotWin(master=mainwin, plot=fftimage, norm=LogNorm(vmin=5))


def highpass():
    print("highpass")
    f = tk.simpledialog.askfloat("subtrack_background", "Enter sigma (float)",
                                 initialvalue=20)
    img.highpass(f)
    mainwin.update()
    history.add()


def sigma():
    print("sigma")
    g = tk.simpledialog.askfloat("Set Sigma", "Enter sigma (float)",
                                 initialvalue=3)
    img.sigma(g)
    mainwin.update()
    history.add()


def gamma():
    print("gamma")
    g = tk.simpledialog.askfloat("Set Gamma", "Enter gamma (float)",
                                 initialvalue=.8)
    img.gamma(g)
    mainwin.update()
    history.add()


def clip_high():
    print("clip_high")
    f = tk.simpledialog.askfloat("Cut high", "Enter high treshold (float)",
                                 initialvalue=.9)
    img.clip_high(f)
    mainwin.update()
    history.add()


def clip_low():
    print("clip_low")
    f = tk.simpledialog.askfloat("Cut low", "Enter low treshold (float)",
                                 initialvalue=.1)
    img.clip_low(f)
    mainwin.update()
    history.add()


def tres_high():
    print("tres_high")
    f = tk.simpledialog.askfloat("tres high", "Enter high treshold (float)",
                                 initialvalue=.9)
    img.tres_high(f)
    mainwin.update()
    history.add()


def tres_low():
    print("tres_low")
    f = tk.simpledialog.askfloat("tres low", "Enter low treshold (float)",
                                 initialvalue=.1)
    img.tres_low(f)
    mainwin.update()
    history.add()


def crop():
    print(f"{select} crop")
    img.crop()
    mainwin.update()
    history.add()
    select.reset()


def zoom_out():
    print("zoom out")
    if mainwin.zoom < 50:
        mainwin.zoom += zoom_step()
        mainwin.update()
        select.reset()


def zoom_in():
    print("zoom in")
    if mainwin.zoom > 1:
        mainwin.zoom -= zoom_step()
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

    print("quit app")
    root.destroy()


#  ------------------------------------------
#  HISTORY
#  ------------------------------------------


class History():

    def __init__(self):
        self.undo_queue = deque([], SETTINGS["history_steps"])
        self.redo_queue = deque([], SETTINGS["history_steps"])

    def add(self):
        if SETTINGS["history_steps"] > 0:
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

        mainwin.update()

    def reset(self):
        img.arr = img.original.copy()
        mainwin.update()

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
        self.bind("<Key>", lambda event: keyPressed(event))
        self.draw(*a, **kw)

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

    def __init__(self, master=None, linewidth=1.0):
        super().__init__(master)
        self.title("Numpyshop-histogram")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", hist_toggle)
        self.geometry("300x300")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.linewidth = linewidth
        self.bins = SETTINGS["histogram_bins"]

        self.hidden = SETTINGS["hide_histogram"]
        if self.hidden:
            self.withdraw()

        self.draw()

#    @timeit
    def draw(self):
        self.fig = plt.figure(figsize=(2, 4))
        self.axes = self.fig.add_subplot(111)

        plt.xticks(np.linspace(0, 1, 11), rotation='vertical')
        self.ims = []

        x, hist_data = img.histogram_data(bins=self.bins)

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

#    @timeit
    def update(self):

        if self.hidden:
            return

        x, hist_data = img.histogram_data(bins=self.bins)
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

#    @timeit
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

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", quit_app)
        self.geometry("900x810")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.bind("<MouseWheel>", self._on_mousewheel)  # windows
        self.bind("<Button-4>", self._on_mousewheel)  # linux
        self.bind("<Button-5>", self._on_mousewheel)  # linux
        self.bind("<Button-1>", self._on_mouse_left)
        self.bind("<Button-3>", self._on_mouse_right)
        self.zoom = max(1, img.width * img.height // 2**22)
        print(self.zoom)
        if not SETTINGS["hide_toolbar"]:
            self.buttons_init()
        self.menu_init()
        self.canvas_init()

        self.make_image_view()
        self.draw()

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

    def _on_mousewheel(self, event):
        print(event.delta, event.num)
        if event.num == 5 or event.delta == -120:
            zoom_in()
        if event.num == 4 or event.delta == 120:
            zoom_out()

    @timeit
    def make_image_view(self):

        print(img.arr.shape)
        print(self.zoom)

        view = img.arr[::self.zoom, ::self.zoom, ...]
        self.view_shape = view.shape[:2]

        view = img_as_ubyte(view)
        view = Image.fromarray(view)
        self.view = ImageTk.PhotoImage(view, master=self)

    @timeit
    def draw(self):
        ''' draw new image '''
        self.make_image_view()
        self.image = self.canvas.create_image(0, 0,
                                              anchor="nw", image=self.view)

    @timeit
    def update(self):
        ''' update image '''
        self.draw()

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

    def __init__(self, parent):
        self.parent = parent
        self.geometry = [0, 0, 0, 0]
        self.rect = None

    def image_selected(self):
        ''' recalculate selection by zoom '''
        x0, y0, x1, y1 = [mainwin.zoom * c for c in self.geometry]
        img.slice = np.s_[y0:y1, x0:x1, ...]

    def set_left(self):
        self.geometry[:2] = list(get_mouse())
        self.draw()

    def set_right(self):
        self.geometry[2:] = list(get_mouse())
        self.draw()

    def draw(self):
        self._valid_selection()
        print(self.geometry)
        mainwin.canvas.delete(self.rect)
        self.rect = mainwin.canvas.create_rectangle(self.geometry)
        print(self.rect)
        self.image_selected()
        histwin.update()

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

    def __str__(self):
        return f"selection geom: {self.geometry}"
#  ------------------------------------------
#  MAIN
#  ------------------------------------------


if __name__ == '__main__':

    if len(sys.argv) > 1:
        Fp = Path(sys.argv[1])
        assert Fp.is_file(), f"not a file {Fp}"
    else:
        Fp = Path(__file__).parent / 'sample.jpg'

    root = tk.Tk()
    root.title("Numpyshop")
    root.withdraw()  # root win is hidden

    # load image into numpy array
    img = npImage(Fp)
    print("image loaded")

    select = Selection(root)
    print(select)
    select.set_left
    print(select)

    history = History()

    histwin = histWin(root)

    statswin = statsWin(root)

    mainwin = mainWin(root)

    print(f"mainloop in {time.time()-time0}")

    root.mainloop()
