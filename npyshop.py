#!/usr/bin/env python3
from testing.timeit import timeit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from skimage import img_as_ubyte
from matplotlib import pyplot as plt
from npimage import npImage
from nphistory import History
import nputils
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
    "hide_histogram": False,
    "hide_toolbar": True,
    "hide_stats": True,
    "histogram_bins": 256,
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
                ("Reset", "Q", original),
            ],
        "Edit":
            [
                ("Undo", "z", undo),
                ("Redo", "y", redo),
                ("Crop", "C", crop),
                ("Select left corner", "comma", select.set_left),
                ("Select right corner", "period", select.set_right),
                ("Rotate_90", "r", rotate_90),
                ("Rotate_270", "R", rotate_270),
                ("Rotate_180", "u", rotate_180),
                ("Free Rotate", "f", free_rotate),
            ],
        "Filter":
            [
                ("Gamma", "g", gamma),
                ("Normalize (BW only)", "n", normalize),
                ("Multiply", "m", multiply),
                ("Contrast", "c", contrast),
                ("Add", "a", add),
                ("Invert", "i", invert),
                ("Sigma", "I", sigma),
                ("Unsharp mask", "M", unsharp_mask),
                ("Blur", "B", blur),
                ("Highpass", "H", highpass),
                ("Clip light (BW only)", "l", clip_high),
                ("Clip dark (BW only)", "d", clip_low),
                ("Tres light (BW only)", "L", tres_high),
                ("Tres dark (BW only)", "D", tres_low),
                ("FFT", "F", fft),
                ("iFFT", "Control-F", ifft),
                ("rgb2gray", "b", rgb2gray),
                ("delete", "Delete", delete),
                ("delete c", "Control-Delete", delete_cirk),
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
#  FUNCTIONS
#  ------------------------------------------

def load(fp=""):
    print("open")
    img.load(fp)
    history.original = img.arr
    print(history.original)
    mainwin.title(img.fpath)
    mainwin.reset()
    histwin.reset()


def save():
    print("save")
    img.save()


def save_as():
    print("save as")
    img.save_as()
    mainwin.title(img.fpath)
    mainwin.title(img.fpath)


def original():
    print("toggle original")
#    img.reset()
    if not history.toggle_original:
        img.arr = history.original
        history.toggle_original = not history.toggle_original
        mainwin.update()
    else:
        if history.last() is not None:
            img.arr = history.last()
            history.toggle_original = not history.toggle_original
            mainwin.update()


def undo():
    print("undo")
    prev_arr = history.undo()
    if prev_arr is not None:
        img.arr = prev_arr
        mainwin.update()


def redo():
    print("redo")
    next_arr = history.redo()
    if next_arr is not None:
        img.arr = next_arr
        mainwin.update()


def free_rotate():
    print("free rotate")
    f = tk.simpledialog.askfloat("Rotate", "Angle (float - clockwise)",
                                 initialvalue=2.)
    img.free_rotate(-f)  # clockwise
    mainwin.update()
    history.add(img.arr)
    select.reset()


def rotate_90():
    print("rotate")
    img.rotate()
    mainwin.update()
    history.add(img.arr)
    select.reset()


def rotate_270():
    print("rotate left")
    img.rotate(3)
    mainwin.update()
    history.add(img.arr)
    select.reset()


def rotate_180():
    print("rotate 180")
    img.rotate(2)
    mainwin.update()
    history.add(img.arr)
    select.reset()


def invert():
    print("invert")
    img.invert()
    mainwin.update()
    history.add(img.arr)


def mirror():
    print("mirror")
    img.mirror()
    mainwin.update()
    history.add(img.arr)


def flip():
    print("flip")
    img.flip()
    mainwin.update()
    history.add(img.arr)


def contrast():
    print("contrast")
    f = tk.simpledialog.askfloat("contrast", "Value to multiply with (float)",
                                 initialvalue=1.3)
    img.contrast(f)
    mainwin.update()
    history.add(img.arr)


def multiply():
    print("multiply")
    f = tk.simpledialog.askfloat("Multiply", "Value to multiply with (float)",
                                 initialvalue=1.3)
    img.multiply(f)
    mainwin.update()
    history.add(img.arr)


def add():
    print("add")
    f = tk.simpledialog.askfloat("Add", "Enter value to add (float)",
                                 initialvalue=.2)
    img.add(f)
    mainwin.update()
    history.add(img.arr)


def normalize():
    print("normalize")
    img.normalize()
    mainwin.update()
    history.add(img.arr)


def fill():
    print("fill")
    f = tk.simpledialog.askfloat("Fill", "Value to fill (float)",
                                 initialvalue=0)
    img.fill(f)
    mainwin.update()
    history.add(img.arr)


def delete():
    print("delete")
    img.fill(1)
    mainwin.update()
    history.add(img.arr)

def delete_cirk():
    print("delete")
#    print(select.cirk_mask)
    img.arr = img.arr*(1-select.cirk_mask())
    mainwin.update()
    history.add(img.arr)




def rgb2gray():
    print("rgb2gray")
    img.rgb2gray()
    mainwin.update()
    histwin.reset()
    history.add(img.arr)


def fft():
#    from matplotlib.colors import LogNorm
    print("fft")
    fftimage = img.fft()
#    plotWin(master=mainwin, plot=fftimage, norm=LogNorm(vmin=5))
    plotWin(master=mainwin, plot=fftimage)


def ifft():
    print("ifft")
    im = img.ifft()
    plotWin(master=mainwin, plot=im)


def unsharp_mask():
    print("unsharp_mask")
    r = tk.simpledialog.askfloat("unsharp_mask", "Enter radius (float)",
                                 initialvalue=.5)
    a = tk.simpledialog.askfloat("unsharp_mask", "Enter amount (float)",
                                 initialvalue=0.2)
    img.unsharp_mask(r,a)
    mainwin.update()
    history.add(img.arr)



def blur():
    print("blur")
    f = tk.simpledialog.askfloat("blur", "Enter radius (float)",
                                 initialvalue=1)
    img.blur(f)
    mainwin.update()
    history.add(img.arr)


def highpass():
    print("highpass")
    f = tk.simpledialog.askfloat("subtrack_background", "Enter sigma (float)",
                                 initialvalue=20)
    img.highpass(f)
    mainwin.update()
    history.add(img.arr)


def sigma():
    print("sigma")
    g = tk.simpledialog.askfloat("Set Sigma", "Enter sigma (float)",
                                 initialvalue=3)
    img.sigma(g)
    mainwin.update()
    history.add(img.arr)


def gamma():
    print("gamma")
    g = tk.simpledialog.askfloat("Set Gamma", "Enter gamma (float)",
                                 initialvalue=.8)
    img.gamma(g)
    mainwin.update()
    history.add(img.arr)


def clip_high():
    print("clip_high")
    f = tk.simpledialog.askfloat("Cut high", "Enter high treshold (float)",
                                 initialvalue=.9)
    img.clip_high(f)
    mainwin.update()
    history.add(img.arr)


def clip_low():
    print("clip_low")
    f = tk.simpledialog.askfloat("Cut low", "Enter low treshold (float)",
                                 initialvalue=.1)
    img.clip_low(f)
    mainwin.update()
    history.add(img.arr)


def tres_high():
    print("tres_high")
    f = tk.simpledialog.askfloat("tres high", "Enter high treshold (float)",
                                 initialvalue=.9)
    img.tres_high(f)
    mainwin.update()
    history.add(img.arr)


def tres_low():
    print("tres_low")
    f = tk.simpledialog.askfloat("tres low", "Enter low treshold (float)",
                                 initialvalue=.1)
    img.tres_low(f)
    mainwin.update()
    history.add(img.arr)


def crop():
    print(f"{select} crop")

    img.crop(*select.geometry)
    mainwin.update()
    history.add(img.arr)
    select.reset()


def zoom_out():
    print("zoom out")
    if mainwin.zoom < 50:
        mainwin.zoom += zoom_step()
        mainwin.update()
        select.reset()


def circular_mask():
    print("zoom in")
    select.make_cirk_mask()

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

    def draw(self):
        # empty graph
        self.fig, self.ax = plt.subplots()
        self.ax.set_xticks(np.linspace(0, 1, 11))
        self.ax.set_ylim(0, 10)
        self.ax.set_title("Histogram")
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.tick_params(left=False)
        self.fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.data = {'black':None, 'red':None, 'green':None, 'blue':None,}

        # get data
        self.x, hist_data = img.histogram_data(bins=self.bins)

        for color in self.data:
            self.data[color] = self.ax.plot(self.x, 0*self.x, color=color)[0]
        self.update()

    def reset(self):

        for color in self.data:
            self.data[color].set_data(self.x,0*self.x)

        self.update()


    def update(self):
        if self.hidden:
            return
        x, hist_data = img.histogram_data(bins=self.bins)
        # loop and update all lines
        for color, y in hist_data.items():
            self.data[color].set_data(x,y)

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
        self.master = master
        self.protocol("WM_DELETE_WINDOW", quit_app)
        self.geometry("900x810")
        self.bind("<Key>", lambda event: keyPressed(event))
        self.bind("<MouseWheel>", self._on_mousewheel)  # windows
        self.bind("<Button-4>", self._on_mousewheel)  # linux
        self.bind("<Button-5>", self._on_mousewheel)  # linux
        self.bind("<Button-1>", self._on_mouse_left)
        self.bind("<Button-3>", self._on_mouse_right)
        self.title(img.fpath)
        self.zoom = 1
        if not SETTINGS["hide_toolbar"]:
            self.buttons_init()
        self.menu_init()
        self.canvas_init()
        self.reset()

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

    def reset(self):
        self.zoom = max(1, img.width * img.height // 2**22)
        print(f"initial zoom set: {self.zoom}")
        self.update()



    @timeit
    def update(self):
        ''' update image '''
        self.draw()
        self.title(img.properties)
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
        self.mask = None
        self.cirk_mode = False

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
#        print(self.geometry)
        mainwin.canvas.delete(self.rect)
        self.rect = mainwin.canvas.create_rectangle(self.geometry)
#        print(self.rect)
        self.image_selected()
        if self.cirk_mode:
            self.make_cirk_mask()
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


    def make_cirk_mask(self):
        x0, y0, x1, y1 = self.geometry
        b, a = (x1+x0)/2, (y1+y0)/2

        nx,ny = img.arr.shape[:2]
        y,x = np.ogrid[-a:nx-a,-b:ny-b]
        print(y,x)
        radius = x0-b
        print(radius)
        mask = x*x + y*y <= radius*radius

        return mask

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

    history = History(max_length=SETTINGS["history_steps"])

    # load image into numpy array
    img = npImage(Fp)
    history.original = img.arr

    print("image loaded")

    select = Selection(root)
    print(select)
    select.set_left
    print(select)



    histwin = histWin(root)

    statswin = statsWin(root)

    mainwin = mainWin(root)

    print(f"mainloop in {time.time()-time0}")

    root.mainloop()
