#!/usr/bin/env python3
import tkinter as tk
from testing.timeit import timeit

#  ------------------------------------------
#  STATISTICS
#  ------------------------------------------


class statsWin(tk.Toplevel):
    @timeit
    def __init__(self, hide=True, master=None):
        super().__init__(master)
        self.title("Stats")
        self.master = master
#        self.protocol("WM_DELETE_WINDOW", stats_toggle)
        self.geometry("150x230")
#        self.bind("<Key>", lambda event: keyPressed(event))
        self.hidden = hide
        self.frame = tk.Frame(self)
        if self.hidden:
            self.withdraw()


    @timeit
    def update(self):

        if self.hidden:
            return
        self.frame.grid_forget()
        self._draw_table()

#    @timeit
    def _draw_table(self):

        for r, k in enumerate(self.master.img.stats):  # loop stats dictionary
            bg = "#ffffff" if r % 2 else "#ddffee"  # alternating row colors
            # keys
            b1 = tk.Label(self.frame, text=k, font=(None, 9),
                          background=bg, width=9)
            b1.grid(row=r, column=1)

            # values
            b2 = tk.Label(self.frame, text=self.master.img.stats[k], font=(None, 9),
                          background=bg, width=9)
            b2.grid(row=r, column=2)

        self.frame.pack(side=tk.LEFT)

