import os
import shutil


ROOT_DIR = "16000_pcm_speeches"
AUDIO_DIR = "audio"

NEW_ROOT_DIR = "500_samples"
AMOUNT_TO_COPY = 500

INPUT_PATH = os.path.join(ROOT_DIR, AUDIO_DIR)
OUTPUT_PATH = os.path.join(NEW_ROOT_DIR, AUDIO_DIR)

DIR_NAMES = os.listdir(INPUT_PATH)
print(DIR_NAMES)

for NAME in DIR_NAMES:
    path = os.path.join(INPUT_PATH,NAME)
    output_path = os.path.join(OUTPUT_PATH,NAME)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for i in range(AMOUNT_TO_COPY):
        clip = os.path.join(path,f'{i}.wav')
        shutil.copy(clip, output_path)
        