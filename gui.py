from tkinter.font import families
import tkinter as tk
import sklearn as sk
import pandas as pd
import numpy as np
import pickle
import tkinter.font as tkFont
from tkinter import ttk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

def sanitize(text):
    out = ''
    text = " ".join(text.casefold().split())
    for t in text:
        if 96 < ord(t) < 123 or t in (" ", ","):
            out += t
    return out

def preprocess(row):
    if isinstance(row, pandas.Series):
        combined_text = " ".join([sanitize(row[k]) for k in ('language', 'origin_query', 'category_path')])
    elif isinstance(row, str):
        combined_text = sanitize(row)
    elif isinstance(row, list):
        combined_text = " ".join([sanitize(row[k]) for k in range(3)])
    else:
        raise TypeError("Input of invalid type.")
    return combined_text

class ProductionClassifier:
    def __init__(self, name):
        self.vect = TfidfVectorizer(ngram_range=(1,2))
        self.cls = RandomForestClassifier(n_estimators=100,random_state=42,
                                          min_samples_split= 20,n_jobs=-1 ,verbose=3)
        self.trained = 0

        try:
            with open(name, "rb") as f:
                dic = pickle.load(f)
            if dic["trained"] !=1:
                raise ValueError
            self.vect = dic["vect"]
            self.cls = dic["cls"]
        except Exception:
            print("Model not found.")

    def predict(self, rw):
        prepared_text = preprocess(rw)
        fv = self.vect.transform([prepared_text])
        p = self.cls.predict(fv)
        return p[0]

class MainScreen:
    def __init__(self):
        global theme, fontpack
        self.themename = "Dark"
        theme = themes[self.themename]
        fontpack = ['Calibri', 18]
        self.root = tk.Tk()
        self.updatefont()
        self.root.title('AI Query')
        self.root.geometry('700x500')
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)

        self.fontprompt = Prompt(self.customfont, 0)
        self.fontprompt.cur = fontpack[0]
        self.fontprompt.screen.title("Enter Font Name")
        self.fontprompt.ent = tk.Entry(self.fontprompt.screen)
        self.fontprompt.button = tk.Button(self.fontprompt.screen, text="Go", command=lambda: self.changefont(custom=True))

        self.cls = ProductionClassifier("models/rfcOPT.bin")
        self.index = 0
        self.query = ''
        self.hier = ''
        self.lang = ''

        self.addmenubar()
        self.changetheme(self.themename)
        self.tk_display()
        self.root.mainloop()

    def promptload(self, widgets=[], padx=[], pady=[], ok=0):
        for i in range(len(widgets)):
            widgets[i].pack(padx=padx[i], pady=pady[i])
        if ok != 0:
            tk.Button(ok.screen, text='Ok', command=ok.close, bg=theme[3], fg=theme[4]).pack(pady=10)

    def changetheme(self, name='Dark'):
        global theme
        self.preferencemenu.entryconfigure(2, label=f'Theme: {name}')
        self.themename = name
        theme = themes[name]
        self.menubar.config(bg = theme[3], fg=theme[4])
        self.preferencemenu.config(bg=theme[3], fg=theme[4])
        self.fontoption.config(bg=theme[3], fg=theme[4])
        self.fontsizeoption.config(bg=theme[3], fg=theme[4])
        self.thememenu.config(bg=theme[3], fg=theme[4])
        self.update_children(self.root)

    def updatefont(self):
        tkFont.nametofont("TkDefaultFont").configure(family=fontpack[0], size=fontpack[1])
        tkFont.nametofont("TkTextFont").configure(family=fontpack[0], size=fontpack[1])
        tkFont.nametofont("TkMenuFont").configure(family=fontpack[0], size=fontpack[1])
        tkFont.nametofont("TkHeadingFont").configure(family=fontpack[0], size=fontpack[1])

    def update_children(self, window):
        window.config(bg=theme[2])
        for w in window.winfo_children():
            if isinstance(w, tk.Entry) or isinstance(w, tk.OptionMenu):
                w.config(fg=theme[0], bg=theme[1])
            elif isinstance(w, tk.Button):
                w.config(fg=theme[4], bg=theme[3])# 01728725
            elif isinstance(w, tk.Text):
                w.config(fg=theme[4], bg=theme[3])
            elif isinstance(w, tk.Label):
                if w in self.errormessages:
                    w.config(fg=theme[5], bg=theme[2])
                else:
                    w.config(bg=theme[2], fg=theme[0])
            elif isinstance(w, tk.Frame) or isinstance(w, tk.Canvas) or isinstance(w, tk.Toplevel):
                self.update_children(w)

    def changefont(self, font=False, size=False, custom=False):
        if font:
            fontpack[0] = font
            self.preferencemenu.entryconfigure(0, label=f'Font: {font}')
        if size:
            fontpack[1] = size
            self.preferencemenu.entryconfigure(1, label=f'Font: {size}')
        if custom:
            font = self.fontprompt.ent.get().strip()
            self.fontprompt.cur = font
            fontpack[0] = font
            self.fontoption.entryconfigure(19, label=f"Custom: {font}")
            self.preferencemenu.entryconfigure(0, label=f'Font: {font}')
        self.updatefont()
        self.fontprompt.close()

    def customfont(self):
        self.fontprompt.ent.delete(0, tk.END)
        self.fontprompt.ent.insert(0, self.fontprompt.cur)
        self.fontprompt.ent.pack(padx=10)
        self.fontprompt.button.pack(pady=5)

    def addmenubar(self):
        self.menubar = tk.Menu(self.root)
        self.preferencemenu = tk.Menu(self.menubar, tearoff=0)

        self.fontoption = tk.Menu(self.preferencemenu, tearoff=0)
        self.thememenu = tk.Menu(self.preferencemenu, tearoff=0)
        self.fonts = ['Calibri', 'Helvetica', 'Laksaman', 'Latin Modern Math', 'Latin Modern Mono',
                      'Latin Modern Mono Prop', 'Latin Modern Mono Slanted', 'Latin Modern Roman', 'Latin Modern Roman',
                      'MathJax_Caligraphic', 'MathJax_Main', 'MathJax_Math', 'Nimbus Roman', 'Purisa',
                      'Standard Symbols PS', 'TeX Gyre Pagella', 'Times New Roman', 'URW Gothic', 'URW Palladio L']
        for f in self.fonts:
            self.fontoption.add_command(label=f, command=lambda f=f: self.changefont(f), font=(f, fontpack[1]))
        self.fontoption.add_command(label=f'Custom: {fontpack[0]}', command=self.fontprompt.open, font=fontpack)
        self.fontsizeoption = tk.Menu(self.preferencemenu, tearoff=0)
        for i in range(10, 45, 2):
            self.fontsizeoption.add_command(label=f'{i}', command=lambda i=i: self.changefont(size=i))
        for t in list(themes.keys()):
            self.thememenu.add_command(label=t, command=lambda t=t: self.changetheme(t))
        self.preferencemenu.add_cascade(menu=self.fontoption, label=f'Font: {fontpack[0]}')
        self.preferencemenu.add_cascade(menu=self.fontsizeoption, label=f'Size: {fontpack[1]}')
        self.preferencemenu.add_cascade(menu=self.thememenu, label=f'Theme: {self.themename}')
        self.menubar.add_cascade(menu=self.preferencemenu, label="Preferences")

        self.menubar.add_command(label="Exit", command=self.root.quit)
        self.root.config(menu=self.menubar)

    def tk_display(self):
        self.mainframe = tk.Frame(self.root, bg=theme[2])

        self.buttonframe = tk.Frame(self.mainframe, bg=theme[2])
        self.enterbutton = tk.Button(self.buttonframe, text='Enter', bg=theme[3], fg=theme[4], command = self.execute)
        self.enterbutton.grid(row=0, column=0, sticky='news')

        self.comentry = tk.Entry(self.mainframe, fg=theme[0], bg=theme[1], width=35)
        self.tbox = tk.Text(self.mainframe, fg=theme[0], bg=theme[1])
        self.mainframe.pack(fill=tk.BOTH, expand=1)
        self.buttonframe.grid(row=0, column=0, sticky='news')
        self.comentry.grid(row=1, column=0, sticky='news')
        self.tbox.grid(row=2, column=0)
        self.tbox.insert(tk.END, f'Enter language(en/it/fr...): ')
        self.tbox.config(state=tk.DISABLED)

    def execute(self):
        if self.index == 0:
            self.lang = self.comentry.get().strip()
            if len(self.lang) == 0:
                return
            self.comentry.delete(0, tk.END)
            self.tbox.config(state=tk.NORMAL)
            self.tbox.insert(tk.END, f'{self.lang}\n')
            self.tbox.insert(tk.END, f'Enter query: ')
            self.tbox.config(state=tk.DISABLED)
            self.index = 1
            return
        elif self.index == 1:
            self.query = self.comentry.get().strip()
            if len(self.query) == 0:
                return
            self.comentry.delete(0, tk.END)
            self.tbox.config(state=tk.NORMAL)
            self.tbox.insert(tk.END, f'{self.query}\n')
            self.tbox.insert(tk.END, f'Enter hierarchy: ')
            self.tbox.config(state=tk.DISABLED)
            self.index = 2
            return
        else:
            self.hier = self.comentry.get().strip()
            if len(self.hier) == 0:
                return
            self.comentry.delete(0, tk.END)
            self.tbox.config(state=tk.NORMAL)
            self.tbox.insert(tk.END, f'{self.hier}\n')
            self.index = 0
        try:
            if self.cls.predict([self.lang ,self.query,self.hier]):
                self.tbox.insert(tk.END, f'Yes, the given product is of the given hierarchy (label = 1)\n')
            else:
                self.tbox.insert(tk.END, f'No, the given product does not have the given hierarchy (label = 0)\n')
            self.query = ''
            self.hier = ''
            self.lang = ''
            self.tbox.insert(tk.END, f'Enter language(en/it/fr...): ')
        except:
            self.tbox.insert(tk.END, 'An error unexpected occurred.\n')
        self.tbox.config(state=tk.DISABLED)

class Prompt:
    active = []
    def __init__(self, func=lambda: None, uptime=10, closefunc=0):
        self.exists = False
        self.uptime = uptime*1000
        self.action = func
        self.closef = closefunc
        self.screen = tk.Toplevel()
        self.screen.resizable(0, 0)
        self.screen.protocol("WM_DELETE_WINDOW", self.close)
        self.screen.withdraw()

    def open(self, *args):
        Prompt.active.append(self)
        if self.exists:
            return
        self.exists = True
        self.screen.deiconify()
        self.screen.attributes('-topmost', True)
        if len(args) != 0:
            self.action(args)
        else:
            self.action()
        if self.uptime:
            self.stopseq = self.screen.after(self.uptime, self.close)

    def close(self):
        if self in Prompt.active:
            Prompt.active.remove(self)
        self.exists = False
        for w in self.screen.winfo_children():
            w.pack_forget()
        if self.uptime:
            self.screen.after_cancel(self.stopseq)
        if self.closef != 0:
            self.closef()
        else:
            self.screen.withdraw()


Dark = ['#ffffff', '#52575a', '#2b2d2e', '#1a1a1a', '#ffffff', '#ff0000']
Light = ['#17191a', '#dbdbde', '#c3c3c3', '#a6a6a6', '#17191a', '#ff0000']
Violet = ['#0e0026', '#b990ff', '#5b00ff', '#2e027b', '#cbb9ea', '#ff0000']
Matte = ['#222222', '#ffffff', '#bfbfbf', '#bfbfbf', '#222222', '#ff0000']
themes = {'Dark':Dark, 'Light':Light, 'Violet':Violet}
theme = Dark

MainScreen()
