import os
import tensorflow as tf
from tensorflow import keras
from tkinter import *
import numpy as np

import pyaudio

SAMPLING_RATE = 16000
MODEL_OPTIONS = ["500_samples", "1500_samples"]


class ModelGUI():


    def __init__(self):

        self.root = Tk()
        self.root.title("Test")
        self.root.geometry("2560x1664")

        self.model_name = "500_samples"
        self.model = keras.models.load_model(f"{self.model_name}.keras")
        self.class_names = os.listdir("500_samples/audio")

        self.p = pyaudio.PyAudio()
        self.input_stream = self.p.open(
            format = pyaudio.paFloat32, 
            rate=SAMPLING_RATE, channels =1, 
            input=True, 
            frames_per_buffer=SAMPLING_RATE
        )


        
        
        self.frame = Frame(self.root)
        self.frame.pack()

        self.prediction_labels = self.create_labels()
    
        self.model_label = Label(self.frame, text=self.model_name, font=("Arial",16))
        self.model_label.pack(pady=10)
        

        self.button_frame = Frame(self.frame)
        self.button_frame.pack(pady=20)
        self.create_model_buttons()
        

    def create_labels(self):
        prediction_labels =[]
        for i in range(3):
            label = Label(
                self.frame,
                text="", 
                font=("Arial",30)
            )
            label.pack(pady=20)
            prediction_labels.append(label)
        return prediction_labels

    def create_model_buttons(self):
        for name in MODEL_OPTIONS:
            btn = Button(
                self.button_frame,
                text=name,
                font=("Arial",14),
                width=12,
                command=lambda: self.change_model(name)
            )
            btn.pack(side=LEFT,padx=10)



    def change_model(self,model_name):
        try:
            self.model = keras.models.load_model(f'{model_name}.keras')
            self.model_label.config(text= model_name)
            self.model_name = model_name
            print(f'Model changed to {model_name}')
        except Exception as e:
            print(f'Failed to load model {model_name}')
  



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
        


        probs = pred * 100  
        sorted_idx = np.argsort(probs)[::-1]
        self.display_prediction(sorted_idx)
        
        for idx in sorted_idx:
            print(f"{self.class_names[idx]}: {probs[idx]:.2f}%")



    def display_prediction(self,sorted_idx):
        print("Sannolikheter:")
        
        for i in range(3):
            self.prediction_labels[i].config(text=f'{i+1}. {self.class_names[sorted_idx[i]]}')
        
        #start the prediction loop again
        self.root.after(1000, self.read_from_input_stream)

    def main(self):
        self.root.after(1000, gui.read_from_input_stream)
        self.root.mainloop()


    def destroy(self):
        self.input_stream.stop_stream()
        self.input_stream.close()
        self.p.terminate()
        self.root.destroy()


if __name__ == "__main__":
 
    gui = ModelGUI()
    try:
        gui.main()
    except KeyboardInterrupt:
        gui.destroy()

