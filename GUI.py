import os
import tensorflow as tf
from tensorflow import keras
from tkinter import *

import pyaudio

root = Tk()
root.geometry("200x150")
frame = Frame(root)
frame.pack()


label = Label(frame, text = "hello world!")
label.pack()

root.title("Test")
root.mainloop()


mic_stream = pyaudio.PyAudio.open(rate=16000, channels =1, input=True, frames_per_buffer=16000)

mic_stream.close()


