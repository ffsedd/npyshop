#!/usr/bin/env python3
import tkinter as tk
from tkinter import simpledialog
import os


class dialogException(Exception):
    pass

class askInput(simpledialog._QueryString):
      
        
    def body(self, master):
        self.geometry("+%d+%d" % (500, 500))  # not working
        self.bind('<KP_Enter>', self.ok) # Enter or KP_Enter
        super().body(master)

        
def askfloat(prompt, **kw):
    d = askInput("AskFloat", prompt, **kw).result
    if d is None:
        raise dialogException("Exception: input empty")   
    return float(d)
    
def askint(title, prompt, **kw):
    d = askInput("AskInt", prompt, **kw).result
    if d is None:
        raise dialogException("Exception: input empty")   
    return int(d)
    
def ask(title, prompt, **kw):
    d = askInput("Ask", prompt, **kw).result
    if d is None:
        raise dialogException("Exception: input empty")   
    return str(d)

def askfloat(prompt, **kw):
    d = InputBox(prompt, title="AskFloat", **kw)
    print("return",d,d.result)
    if d.result is None:
        raise dialogException("Exception: input empty")   
    return float(d.result)



class InputBox:
    def __init__(self, prompt='', title='Inputbox', parent=None, initialvalue=''):
        self.win = tk.Toplevel()
           
        self.win.protocol("WM_DELETE_WINDOW", self._quit)
        self.win.parent = parent or tk._default_root
        self.win.title(title)
        self.win.geometry("+%d+%d" % (self.win.parent.winfo_screenwidth()//2,
                                  self.win.parent.winfo_screenheight()//2))
        self.result = None
                     
        tk.Label(self.win, text=prompt).grid(row=0)
        self.win.e1 = tk.Entry(self.win)
        self.win.e1.insert(0, initialvalue)
        self.win.e1.grid(row=0, column=1)
        self.win.e1.focus()

        tk.Button(self.win, text='Cancel', command=self.cancel).grid(row=3, column=1)
        tk.Button(self.win, text='OK', command=self.ok).grid(row=3, column=0)
                                                   
        self.win.bind('<KP_Enter>', lambda event: self.ok())   
        self.win.bind('<Return>', lambda event: self.ok()) 
        self.win.bind("<Escape>", lambda event: self.cancel())
        print("enter mainloop")
        # self.grab_set()
        self.win.mainloop()
        self.win.destroy()
        
                                                 
    def ok(self):
        self.result = self.win.e1.get()
        print("ok")
        
        self._quit()

        
    def cancel(self):
        print("cancel")
        self._quit()

    def _quit(self):
        print("quit inputbox")
        self.win.quit()



if __name__ == "__main__":
 
    root = tk.Tk()
    dialog = InputBox("value", parent=root)
    root.mainloop()
