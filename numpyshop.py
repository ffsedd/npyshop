#!/usr/bin/env python3
import sys
from pathlib import Path

import tkinter as tk
import numpy as np
from npimage import npImage 

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import pyplot as plt





def undo():
    print("undo")
    
def redo():
    print("redo")
    

def keyPressed(event):
    
    commands = [
            ("z",undo),
            ("y",redo),
            ("h",histwin.toggle),
            ("r",toolbar.rotate),
            ("o",toolbar.load),
            ("s",toolbar.save),
            ("g",toolbar.gamma),
            
            ]
    for c in commands:
        if event.keysym == c[0]:
            c[1]()

        

class Toolbar(tk.Toplevel):

    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop-toolbar")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", quit_app)
        self.geometry("600x30")
        self.bind("<Key>", lambda event:keyPressed(event))
        self.ButtonsInit()
        self.menuInit()
        

            
    def ButtonsInit(self):

        backgroundColour="white"
        buttonWidth=6
        buttonHeight=1
        toolKitFrame=tk.Frame(self)

        buttons_cfg = [
                        ("Crop", self.crop),
#                        ("Brightness", self.brightness),
#                        ("Mirror", self.mirror),
#                        ("Flip", self.flip),
                        ("Rotate", self.rotate),
#                        ("LosslessRotate", self.lossless_rotate),
#                        ("Reset", self.reset),
                ]

        for i, b in enumerate(buttons_cfg):
            button=tk.Button(toolKitFrame, text=b[0],\
                              background=backgroundColour ,\
                              width=buttonWidth, height=buttonHeight, \
                              command=b[1])
            button.grid(row=0,column=i)

        toolKitFrame.pack(side=tk.LEFT)
        
        
    def menuInit(self):

        menubar=tk.Menu(self)
#        menubar.add_command(label="New", command=lambda:img.open())
        menubar.add_command(label="Open", command=self.load)
        menubar.add_command(label="Save", command=lambda:self.save)
        menubar.add_command(label="Save As", command=lambda:self.save_as)
        
        ## Edit pull-down tk.Menu
        editmenu = tk.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Undo   Z", command=lambda:undo())
        editmenu.add_command(label="Redo   Y", command=lambda:redo())
        editmenu.add_command(label="Crop   C", command=self.crop)
        editmenu.add_command(label="Rotate   R", command=self.rotate)
        menubar.add_cascade(label="Edit", menu=editmenu)
        self.config(menu=menubar)
        
        
        ## View pull-down tk.Menu
        viewmenu = tk.Menu(menubar, tearoff=0)
        viewmenu.add_command(label="Histogram   H", command=histwin.toggle)

        menubar.add_cascade(label="View", menu=viewmenu)
        self.config(menu=menubar)
        
    
        
        ## Filter pull-down tk.Menu
        filtermenu = tk.Menu(menubar, tearoff=0)
        filtermenu.add_command(label="Invert", command=self.invert)
        filtermenu.add_command(label="Gamma", command=self.gamma)
        menubar.add_cascade(label="Filter", menu=filtermenu)
        
        self.config(menu=menubar)
        
        
    def load(self):
        img.load()
        img.info()
        mainwin.update()
        global histwin
        histwin.destroy()
        histwin = histWin(root) # reload graph - new scale if bitdepth different

    def save(self):
        img.save()
        
    def save_as(self):
        img.save_as()
        
    def rotate(self):
        img.rotate()
        mainwin.update()     
        
    def invert(self):
        img.invert()
        mainwin.update() 
        
    def gamma(self):
        g = tk.simpledialog.askstring("Set Gamma", "Value?",
                                parent=self)
        img.gamma(float(g))
        mainwin.update()     
        
    def crop(self):
        x0 = mainwin.xlim[0]
        x1 = mainwin.xlim[1]
        y1 = mainwin.ylim[0]
        y0 = mainwin.ylim[1]
        print(f"call crop: {x0} {x1} {y0} {y1}")
        img.crop(x0,x1,y0,y1)
        mainwin.update()     
        
                 
        
class mainWin(tk.Toplevel):
    
        
    def __init__(self, master=None):
        super().__init__(master)
        self.title("numpyshop-Canvas")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", quit_app)
        self.geometry("800x800")
        self.bind("<Key>", lambda event:keyPressed(event))
        self.graph()

        
    def graph(self):
        
        self.fig = plt.figure(figsize=(5,4))
        self.cmap = "gray" if img.channels == 1 else "jet"
        self.im = plt.imshow(img.arr, cmap=self.cmap) # later use a.set_data(new_data)
        self.ax = plt.gca()
        self.ax.set_xticklabels([]) 
        self.ax.set_yticklabels([])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('draw_event', self.on_draw)


    def update(self):
        ''' update image '''
        if len(img.arr.ravel()) == 0:
            print("array is empty")
            return
        img.info()
        self.im.set_data(img.arr)
        self.canvas.draw()
        histwin.update()

    def on_draw(self, event):
        ''' track current selection coordinates '''
        self.xlim = self.ax.get_xlim()
        self.ylim = self.ax.get_ylim()  
        # print(self.xlim,self.ylim)
        
        
class histWin(tk.Toplevel):
    
    def __init__(self, master=None):
        super().__init__(master)
        self.title("Numpyshop-histogram")
        self.master = master
        self.protocol("WM_DELETE_WINDOW", self.toggle)
        self.geometry("300x300")
        self.bind("<Key>", lambda event:keyPressed(event))
        self.hidden = True
        self.bins = 30
        self.x = np.linspace(0,2 ** img.bitdepth, self.bins)
        
        if self.hidden:
            self.withdraw()
        self.graph()
        
    def on_closing(self):
        self.withdraw()  # hide only
        
    def toggle(self):
        if self.hidden:
            self.deiconify() 
            self.update()

        else:
            self.withdraw()
        self.hidden = not self.hidden
        mainwin.focus_force()
        
    def graph(self):
        
        self.fig = plt.figure(figsize=(2,4))
        self.axes = self.fig.add_subplot(111)
        self.ims = []
        
        for line in self.hist_lines():
            self.ims.append(plt.plot(self.x, line[0], color=line[1])[0])
#        print("self.ims",self.ims)
#        self.axes = plt.axes()
        self.ax = plt.gca()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    
    def update(self):
        if self.hidden:
            return
        
        for im, line in zip(self.ims, self.hist_lines()):
#            print("im line ",im, line)
            im.set_data(self.x,line[0])
        self.axes.set_autoscale_on(True)
        self.axes.autoscale_view(True,True,True)

        self.canvas.draw() 
        
    def plot_hist(self, y, color="black"):
        h = np.histogram(y, bins=self.bins, density=True)[0] # normalized
        return h, color
    
    def hist_lines(self):
        ''' return list of histogram y values (1D) '''
        lines = []
        if img.channels == 1: # gray
            y = img.arr
            lines.append(self.plot_hist(y, color="black"))
        else:    # RGB image
            for i,color in enumerate(('red','green','blue')):
                y = img.arr[:,:,i]
                lines.append(self.plot_hist(y, color=color)) 
        return lines
        
def quit_app():
    print("quit app")
    root.destroy()        
        
    
def ret():
    print("ret")
    

    
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
    
    histwin = histWin(root) 
    
    toolbar = Toolbar(root)
   
    mainwin = mainWin(root)
    

    
    
    root.mainloop()
