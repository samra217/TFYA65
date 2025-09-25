import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

SAMPLING_RATE = 16000

# Reuse the same preprocessing from training
def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

def preprocess_audio(file_path):
    # Load audio
    y, sr = librosa.load(file_path, sr=SAMPLING_RATE)

    # Ensure 1 sec length
    desired_len = SAMPLING_RATE
    if len(y) < desired_len:
        y = np.pad(y, (0, desired_len - len(y)))
    else:
        y = y[:desired_len]

    # Shape -> (samples, 1)
    y = y.reshape(-1, 1)

    # Convert to tensor
    audio_tensor = tf.convert_to_tensor([y], dtype=tf.float32)  # shape (1, 16000, 1)

    # FFT transform
    fft_tensor = audio_to_fft(audio_tensor)  # shape (1, 8000, 1)

    return fft_tensor

# Load your saved model
model = keras.models.load_model("TFYA65.keras")

# Class names (must be in the same order as training!)
class_names = os.listdir("16000_pcm_speeches/audio")

# Pick a test file
test_file = "/Users/samuelswahnrasch/Documents/TFYA65/16000_pcm_speeches/audio/Benjamin_Netanyau/1480.wav"
x = preprocess_audio(test_file)


# Predict
pred = model.predict(x)[0]  # shape (num_classes,)

# Gör om till procent
probs = pred * 100  

# Sortera från högst till lägst
sorted_idx = np.argsort(probs)[::-1]

print("Sannolikheter:")
for idx in sorted_idx:
    print(f"{class_names[idx]}: {probs[idx]:.2f}%")