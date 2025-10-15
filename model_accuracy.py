import tensorflow as tf
from tensorflow import keras
import librosa
import numpy as np
import os
import pathlib as Path

SAMPLING_RATE = 16000

AMOUNT_OF_SAMPLES = 200

DATASET_ROOT = "16000_pcm_speeches"

AUDIO_SUBFOLDER = "audio"
NOISE_SUBFOLDER = "noise"


DATASET_AUDIO_PATH = os.path.join(DATASET_ROOT, AUDIO_SUBFOLDER)
DATASET_NOISE_PATH = os.path.join(DATASET_ROOT, NOISE_SUBFOLDER)

model_names = ["500_samples.keras", "1000_samples.keras", "1250_samples.keras"]

def audio_to_fft(audio):
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

def paths_and_labels_to_dataset(audio_paths, labels):
    """Constructs a dataset of audios and labels."""
    path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
    audio_ds = path_ds.map(
        lambda x: path_to_audio(x), num_parallel_calls=tf.data.AUTOTUNE
    )
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    return tf.data.Dataset.zip((audio_ds, label_ds))


def path_to_audio(path):
    """Reads and decodes an audio file."""
    audio = tf.io.read_file(path)
    audio, _ = tf.audio.decode_wav(audio, 1, SAMPLING_RATE)
    return audio


def audio_to_fft(audio):
    # Since tf.signal.fft applies FFT on the innermost dimension,
    # we need to squeeze the dimensions and then expand them again
    # after FFT
    audio = tf.squeeze(audio, axis=-1)
    fft = tf.signal.fft(
        tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
    )
    fft = tf.expand_dims(fft, axis=-1)

    # Return the absolute value of the first half of the FFT
    # which represents the positive frequencies
    return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])


# Get the list of audio file paths along with their corresponding labels

class_names = os.listdir(DATASET_AUDIO_PATH)
print(
    "Our class names: {}".format(
        class_names,
    )
)



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




audio_paths = []
labels = []
for label, name in enumerate(class_names):
    print(
        "Processing speaker {}".format(
            name,
        )
    )
    dir_path = Path(DATASET_AUDIO_PATH) / name
    

    speaker_sample_paths = [
        os.path.join(dir_path, filepath)
        for filepath in reversed(os.listdir(dir_path))
        if filepath.endswith(".wav")
    ]
    speaker_sample_paths = speaker_sample_paths[:AMOUNT_OF_SAMPLES]

    audio_paths += speaker_sample_paths
    labels += [label] * len(speaker_sample_paths)

audio_paths = audio_paths.map(
    lambda x, y: (audio_to_fft(x), y)
)




for name in model_names:
    model = keras.models.load_model(name)
    print(model.evaluate(audio_paths,labels, verbose = 1))







