import os

from pathlib import Path
from pydub import AudioSegment

CLIP_LENGTH_MS = 1000
SAMPLING_RATE = 16000

ROOT_DIR = "16000_pcm_speeches"
AUDIO_DIR = "audio"
NAME = "Alexander"


OUTPUT_PATH = os.path.join(ROOT_DIR,AUDIO_DIR, NAME)

input = f'{NAME}.wav'


audio = AudioSegment.from_wav(input).set_frame_rate(16000).set_channels(1)



samuel_full_clips = len(audio) // CLIP_LENGTH_MS



for i in range(samuel_full_clips):
    start = i*CLIP_LENGTH_MS
    clip = audio[start:start+CLIP_LENGTH_MS]
    clip.export(os.path.join(OUTPUT_PATH, f'{i}.wav'), format="wav")







