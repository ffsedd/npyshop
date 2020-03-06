#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from subprocess import run
import qq
import logging
import tkinter as tk

from tkinter import Tk, Entry, W, E, N, S, PhotoImage, Checkbutton, Button, \
        Menu, Frame, Label, Spinbox
import logging

PROGRAM_NAME = 'Npyshop'






class Npyshop:

    def __init__(self, root):
        self.root = root
        self.root.title(PROGRAM_NAME)
        self.init_gui()




    def create_right_button_matrix(self):
        right_frame = tk.Frame(self.root)
        right_frame.grid(row=10, column=6, sticky=tk.W + tk.E + tk.N + tk.S, padx=15, pady=4)
        self.buttons = [[None for x in range(
            self.find_number_of_columns())] for x in range(MAX_NUMBER_OF_DRUM_SAMPLES)]
        for row in range(MAX_NUMBER_OF_DRUM_SAMPLES):
            for col in range(self.find_number_of_columns()):
                self.buttons[row][col] = Button(
                    right_frame, command=self.on_button_clicked(row, col))
                self.buttons[row][col].grid(row=row, column=col)
                self.display_button_color(row, col)

    def create_left_drum_loader(self):
        left_frame = tk.Frame(self.root)
        left_frame.grid(row=10, column=0, columnspan=6, sticky=tk.W + tk.E + tk.N + tk.S)
        open_file_icon = PhotoImage(file='images/openfile.gif')
        for i in range(5):
            open_file_button = Button(left_frame, image=open_file_icon,
                                      command=self.on_open_file_button_clicked(i))
            open_file_button.image = open_file_icon
            open_file_button.grid(row=i, column=0,  padx=5, pady=4)
            self.drum_load_entry_widget[i] = Entry(left_frame)
            self.drum_load_entry_widget[i].grid(
                row=i, column=4, padx=7, pady=4)

    def create_top_bar(self):
        topbar_frame = tk.Frame(self.root, height=25)
        topbar_frame.grid(row=0, columnspan=12, rowspan=10, padx=5, pady=5)

        tk.Label(topbar_frame, text='Pattern Number:').grid(row=0, column=1)



    def create_top_menu(self):
        self.menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Load Project")
        self.file_menu.add_command(label="Save Project")
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit")
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        self.about_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.about_menu.add_command(label="About")
        self.menu_bar.add_cascade(label="About", menu=self.about_menu)
        self.root.config(menu=self.menu_bar)

    def init_gui(self):
        self.create_top_menu()
        self.create_top_bar()
        self.create_left_drum_loader()
        self.create_right_button_matrix()


if __name__ == '__main__':
    logging.basicConfig(
#            filename=LOG_FPATH, filemode='w',
        level=20,
        format='%(relativeCreated)d !%(levelno)s [%(module)10s%(lineno)4d]\t%(message)s')

    root = tk.Tk()
    Npyshop(root)
    root.mainloop()