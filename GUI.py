import os
import tensorflow as tf
from tensorflow import keras
from tkinter import *
import numpy as np

import pyaudio

SAMPLING_RATE = 16000
root = Tk()
root.title("Test")

class ModelGUI():


    def __init__(self):

        

        self.model = keras.models.load_model("500_samples.keras")
        self.class_names = os.listdir("500_samples/audio")

        self.p = pyaudio.PyAudio()
        self.input_stream = self.p.open(format = pyaudio.paFloat32, rate=SAMPLING_RATE, channels =1, input=True, frames_per_buffer=SAMPLING_RATE)


        
        root.geometry("2560x1664")
        self.frame = Frame(root)
        self.frame.pack()


        self.label = Label(self.frame)
        self.label.pack()
        

        self.button_frame = Frame(self.frame)
        self.button1 = Button(self.button_frame,text="BigBoy", command=self.change_model("1500_samples")).pack()
        self.button2 = Button(self.button_frame,text="MediumBoy",command=self.change_model("500_samples")).pack()
        self.button_frame.pack()





    def change_model(self,model_name):
        self.model = keras.models.load_model(f'{model_name}.keras')



    def audio_to_fft(self,audio):
        audio = tf.squeeze(audio, axis=-1)
        fft = tf.signal.fft(
            tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
        )
        fft = tf.expand_dims(fft, axis=-1)
        return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


    def read_from_input_stream(self):
        audio_bytes = self.input_stream.read(SAMPLING_RATE, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
        self.preprocess_audio(audio_np)


    def preprocess_audio(self,audio_np):
        audio_np = audio_np.reshape(-1,1)
        audio_tensor = tf.convert_to_tensor([audio_np], dtype=tf.float32)
        fft = self.audio_to_fft(audio_tensor)
        self.predict_speaker(fft)


        
    def predict_speaker(self,fft):
        pred = self.model.predict(fft)[0]
        predicted_speaker_idx = np.argmax(pred)
        self.display_prediction(predicted_speaker_idx)


        probs = pred * 100  
        sorted_idx = np.argsort(probs)[::-1]
        
        for idx in sorted_idx:
            print(f"{self.class_names[idx]}: {probs[idx]:.2f}%")



    def display_prediction(self,speaker_idx):
        print("Sannolikheter:")
        
        self.label['text'] = self.class_names[speaker_idx]
        self.label.pack()



        #start the prediction loop again
        root.after(1000, self.read_from_input_stream)


    def terminate_streams(self):
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.p.terminate()


gui = ModelGUI()
root.after(1000, gui.read_from_input_stream)


root.mainloop()
gui.terminate_streams()
root.destroy()
