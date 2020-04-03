#!/usr/bin/env python3
import tkinter as tk


from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from skimage_exposure import cumulative_distribution  # histogram plotting

from testing.timeit import timeit


"""
HISTOGRAM WINDOW
"""


class histWin(tk.Toplevel):

    @timeit
    def __init__(self, master=None, hide=True, bins=256, linewidth=1.0):
        super().__init__(master)
        self.title("Histogram")
        self.master = master
#        self.protocol("WM_DELETE_WINDOW", hist_toggle)
        self.geometry("300x300")
#        self.bind("<Key>", lambda event: keyPressed(event))
        self.linewidth = linewidth
        self.bins = bins
        self.hidden = hide
        plt.rcParams.update({'font.size': 5})
        self.draw()
        if self.hidden:
            self.withdraw()
        self.master.focus_force() 

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
        # self.ax_hist.tick_params(left=False)
        # self.ax_hist.tick_params(top='off', bottom='on', left='off', right='off', labelleft='off', labelbottom='on')

        # self.ax_hist.hist(app.img.arr.ravel(), bins=self.bins, range=(0, 1),
                          # density=True, histtype='step', color='black')

        # Display cumulative distribution
        self.ax_cdf = self.ax_hist.twinx()
        self.ax_cdf.spines['right'].set_visible(False)
        self.ax_cdf.spines['top'].set_visible(False)
        self.ax_cdf.spines['left'].set_visible(False)
        self.ax_cdf.tick_params(left=False)
        # app.img_cdf, bins = exposure.cumulative_distribution(app.img.arr, self.bins)
        # self.ax_cdf.plot(bins, img_cdf, 'r')
        self.ax_cdf.set_yticks([])

        self.fig.tight_layout(pad=6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
#        self.ax_hist.plot(self.x, 0*self.x, color=color)[0]

    def reset(self):
        self.update


    @timeit
    def update(self):

        if self.hidden:
            return

        self.ax_hist.cla()
        self.ax_cdf.cla()
        self.ax_hist.hist(self.master.img.arr.ravel(), bins=self.bins, range=(0, 1),
                          density=True, histtype='step', color='black')
        img_cdf, bins = cumulative_distribution(self.master.img.arr, self.bins)
        self.ax_cdf.plot(bins, img_cdf, 'r')
        self.canvas.draw()
