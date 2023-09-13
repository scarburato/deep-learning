import time

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
import cv2
from multiprocessing import Pool

root = "data"
output_path = "build/processed"
USE_GRAYSCALE = True
N_JOBS = 6

def audio_to_mel_spectrogram(input_audio_path:str, output_image_path:str, duration=1.5, dpi=1, force=False):
    if not force and os.path.exists(output_image_path):
      return

    # Load the audio file, automatically resampled to
    y, sr = librosa.load(
        input_audio_path,
        #duration=duration,
        mono=True
    )

    # Trim silent parts
    _, index = librosa.effects.trim(y)
    y = y[index[0]:index[1]] # Using directly the function's return doesn't work

    file_duration = librosa.get_duration(y=y, sr=sr)
    if file_duration < duration:
      print(f"Audio duration is shorter than {duration} seconds. Skipping spectrogram generation.")
      return

    chunks = int(file_duration // duration)
    duration_in_samples = librosa.time_to_samples(duration, sr=sr)
    for i in range(0, chunks):
      # cut
      y_cut = y[duration_in_samples*i:duration_in_samples*(i+1)]

      # Compute the Mel spectrogram
      mel_spectrogram = librosa.feature.melspectrogram(y=y_cut, sr=sr)

      # Convert to decibels (log scale)
      mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

      spectrogram_to_png(mel_spectrogram_db, output_image_path, dpi)


def spectrogram_to_png(mel_spectrogram_db, output_image_path: str, dpi):
  assert mel_spectrogram_db.min() >= -80.0 and abs(mel_spectrogram_db.max()) < 1e-3

  mel_spectrogram_db = mel_spectrogram_db * (255.0/-80.0)
  mel_spectrogram_db = np.maximum(0, mel_spectrogram_db).transpose()

  i = Image.fromarray(mel_spectrogram_db.astype(np.uint8), mode = "L")
  i.save(output_image_path)
  i.close()


kraut_emotions_codes = {
  "W": "ANGER",
  "L": "BOREDOM",
  "E": "DISGUST",
  "A": "ANXIETY",
  "F": "HAPPINESS",
  "T": "SADNESS",
  "N": "NEUTRAL"
}

dataset_path = os.path.join(root, "emo-db/")

for key, value in kraut_emotions_codes.items():
  folder_name = os.path.join(output_path, value)
  os.makedirs(folder_name, exist_ok=True)


def job_kraut(audio_file):
  pre, ext = os.path.splitext(audio_file)

  # in this dataset the emotionCode is written inside the name
  emotionCode = pre[5]
  emotion = kraut_emotions_codes[emotionCode]

  out_file =  f"{emotion}/{pre}.png"
  out_path =  os.path.join(output_path, out_file)

  audio_path = os.path.join(dataset_path, audio_file)

  audio_to_mel_spectrogram(
      input_audio_path = audio_path,
      output_image_path = out_path,
  )

with Pool(N_JOBS) as p:
  p.map(job_kraut, os.listdir(dataset_path))


emovo_emotions_codes = {
  "neu": "NEUTRAL",
  "dis": "DISGUST",
  "gio": "HAPPINESS",
  "pau": "ANXIETY",
  "rab": "ANGER",
  "sor": "HAPPINESS",
  "tri": "SADNESS"
}

dataset_path = os.path.join(root, "EMOVO/")


def job_emovo(audio_file):
  emotionCode = audio_file[0:3]

  if emotionCode not in emovo_emotions_codes:
    print("Unknown emotion " + emotionCode)
    return

  pre, ext = os.path.splitext(audio_file)
  out_path = os.path.join(output_path, emovo_emotions_codes[emotionCode], f"emovo_{pre}.png")

  audio_path = os.path.join(folderpath, audio_file)

  audio_to_mel_spectrogram(
    input_audio_path=audio_path,
    output_image_path=out_path,
  )

for actor_folder in ["m3","m2","m1","f3","f2","f1"]:
  folderpath = os.path.join(dataset_path, actor_folder)
  print("Analysis of " + folderpath)

  with Pool(N_JOBS) as p:
    p.map(job_emovo, os.listdir(folderpath))


import gc

emovdb_emotions_codes = {
    "bea_Amused" : "HAPPINESS",
    "bea_Angry" : "ANGER",
    "bea_Disgusted" : "DISGUST",
    "bea_Neutral" : "NEUTRAL",
    "bea_Sleepy" : "BOREDOM",
    "jenie_Amused" : "HAPPINESS",
    "jenie_Angry" : "ANGER",
    "jenie_Disgusted" : "DISGUST",
    "jenie_Neutral" : "NEUTRAL",
    "josh_Amused" : "HAPPINESS",
    "josh_Neutral" : "NEUTRAL",
    "josh_Sleepy" : "BOREDOM",
    "sam_Disgusted" : "DISGUST",
    "sam_Neutral" : "NEUTRAL",
    "sam_Sleepy" : "BOREDOM",
}

dataset_path = os.path.join(root, "EmoV-DB/")


def job_emovdb(audiofilename):
  pre, ext = os.path.splitext(audiofilename)

  # Generate and save to disk
  audio_path = os.path.join(subfolder_name, audiofilename)
  output_filename = os.path.join(output_path, f"{emotion}/{pre}.png")

  audio_to_mel_spectrogram(
    input_audio_path=audio_path,
    output_image_path=output_filename,
  )


for folder_name, emotion in emovdb_emotions_codes.items():
  subfolder_name = os.path.join(dataset_path, folder_name)
  with Pool(N_JOBS) as p:
    p.map(job_emovdb, os.listdir(subfolder_name))


RE_EXTRACT_EMOTION = re.compile(r"\w+_(\w+)_.*_.*\.wav")

# 5 primary emotions: angry, sad, neutral, happy, excited. 5 secondary emotions: anxious, apologetic, pensive, worried, enthusiastic.

jl_emotions_codes = {
  "angry": "ANGER",
  "anxious": "ANXIETY",
  "happy": "HAPPINESS",
  "sad": "SADNESS",
  "neutral": "NEUTRAL",
  "excited": "HAPPINESS", # @TODO DECIDE IF TRUE LOL
  "pensive": "BOREDOM"
}

dataset_path = os.path.join(root, "jl-corpus/")


def job_jl(audio_file):
  pre, ext = os.path.splitext(audio_file)

  # in this dataset the emotionCode is written inside the name
  emotionCode = RE_EXTRACT_EMOTION.findall(audio_file)[0]
  if emotionCode not in jl_emotions_codes:
    print("Unknown emotion " + emotionCode)
    return

  out_path =  os.path.join(output_path, jl_emotions_codes[emotionCode], f"jl_{pre}.png")

  audio_path = os.path.join(dataset_path, audio_file)

  audio_to_mel_spectrogram(
      input_audio_path = audio_path,
      output_image_path = out_path,
  )


with Pool(N_JOBS) as p:
  p.map(job_jl, os.listdir(dataset_path))


# anger disgust fear joy neutral sadness surprise

meld_emotions_codes = {
  "joy": "HAPPINESS",
  "anger": "ANGER",
  "disgust": "DISGUST",
  "neutral": "NEUTRAL",
  "sadness": "SADNESS"
}

dataset_path = os.path.join(root, "MELD.proc/")
RE_EXTRACT_EMOTION_MELD = re.compile(r"\d+_(\w+)_.*")


def job_meld(audio_file):
  pre, ext = os.path.splitext(audio_file)

  # in this dataset the emotionCode is written inside the name
  emotionCode = RE_EXTRACT_EMOTION_MELD.findall(audio_file)[0]
  if emotionCode not in meld_emotions_codes:
    print("Unknown emotion " + emotionCode)
    return

  out_path =  os.path.join(output_path, meld_emotions_codes[emotionCode], f"meld_{pre}.png")

  audio_path = os.path.join(dataset_path, audio_file)

  audio_to_mel_spectrogram(
      input_audio_path = audio_path,
      output_image_path = out_path,
  )

with Pool(N_JOBS) as p:
  p.map(job_meld, os.listdir(dataset_path))


# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Emotional intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.

ravdess_emotions_codes = {
  "05": "ANGER",
  "07": "DISGUST",
  "06": "ANXIETY",
  "03": "HAPPINESS",
  "04": "SADNESS",
  "01": "NEUTRAL"
}

dataset_path = os.path.join(root, "RAVDESS/")
RE_EXTRACT_EMOTION_RAVDESS= re.compile(r"\d+-\d+-(\d+)-.*")


def job_ravdess(audio_file):
  pre, ext = os.path.splitext(audio_file)

  # in this dataset the emotionCode is written inside the name
  emotionCode = RE_EXTRACT_EMOTION_RAVDESS.findall(audio_file)[0]
  if emotionCode not in ravdess_emotions_codes:
    print("Unknown emotion " + emotionCode)
    return

  out_path =  os.path.join(output_path, ravdess_emotions_codes[emotionCode], f"ravdess_{pre}.png")

  audio_path = os.path.join(dataset_path, audio_file)

  audio_to_mel_spectrogram(
      input_audio_path = audio_path,
      output_image_path = out_path,
  )

with Pool(N_JOBS) as p:
  p.map(job_ravdess, os.listdir(dataset_path))

count = {}

for foldername in os.listdir(output_path):
  folderpath = os.path.join(output_path, foldername)
  if not os.path.isdir(folderpath):
    continue

  count[foldername] = 0

  print("Analysis of " + folderpath)
  for imagename in os.listdir(folderpath):
    imagepath = os.path.join(folderpath, imagename)
    pic = cv2.imread(imagepath)
    if pic is None:
      print("#### CORRUPTED #### " + imagepath)
      os.remove(imagepath)
      continue

    count[foldername] += 1

print(count)
