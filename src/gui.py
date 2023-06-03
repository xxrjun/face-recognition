import tkinter as tk
from tkinter.ttk import *

from face_recognition import recognize_image, recognize_video


class GUI:
    def __init__(self, root):
        self.root = root
        self.root.geometry("600x400")
        self.root.title("Face Recognition")
        self.create_buttons()

    def create_buttons(self):
        # define font
        button_font = ('Arial', 18, 'bold')

        # create buttons
        button_img = tk.Button(self.root, text="Recognize Image", width=40, height=3, command=recognize_image, bg='#0052cc', fg='#ffffff')
        button_img['font'] = button_font
        button_img.pack(fill=tk.X, padx=50, pady=10)

        button_vid = tk.Button(self.root, text="Recognize Video", width=40, height=3, command=recognize_video, bg='#0052cc', fg='#ffffff')
        button_vid['font'] = button_font
        button_vid.pack(fill=tk.X, padx=50, pady=10)

        button_quit = tk.Button(self.root, text="QUIT", width=40, height=1, command=self.root.destroy, bg='red', fg='#ffffff')
        button_quit['font'] = button_font
        button_quit.pack(fill=tk.X, padx=50, pady=10)
