import os
import tensorflow as tf
from tensorflow import keras
from tkinter import *
import numpy as np

import pyaudio

SAMPLING_RATE = 16000

RUNNING = True



model = keras.models.load_model("TFYA65.keras")
class_names = os.listdir("16000_pcm_speeches/audio")

p = pyaudio.PyAudio()
root = Tk()
root.geometry("200x150")
frame = Frame(root)
frame.pack()


label = Label(frame, text = "hello world!")
label.pack()


def on_closing():
    root.destroy()
    RUNNING = False
    
root.protocol("WM_DELETE_WINDOW", on_closing)
root.title("Test")




mic_stream = p.open(format = pyaudio.paFloat32, rate=SAMPLING_RATE, channels =1, input=True, frames_per_buffer=SAMPLING_RATE)

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


def read_from_input_stream():
    audio_bytes = mic_stream.read(SAMPLING_RATE, exception_on_overflow=False)
    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
    preprocess_audio(audio_np)

    


def preprocess_audio(audio_np):
    audio_np = audio_np.reshape(-1,1)
    audio_tensor = tf.convert_to_tensor([audio_np], dtype=tf.float32)
    fft = audio_to_fft(audio_tensor)

    pred = model.predict(fft)[0]

    probs = pred * 100  
    sorted_idx = np.argsort(probs)[::-1]

    print("Sannolikheter:")
    for idx in sorted_idx:
        print(f"{class_names[idx]}: {probs[idx]:.2f}%")

    
    root.after(1000, read_from_input_stream)
    

root.after(1000, read_from_input_stream)


root.mainloop()
mic_stream.stop_stream()
mic_stream.close()
p.terminate()


