import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
import cv2

root = "data"
output_path = "build/processed"
USE_GRAYSCALE = True

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
  # Create a Matplotlib figure without axes
  fig, ax = plt.subplots(figsize=mel_spectrogram_db.shape, frameon=False)

  # Display the Mel spectrogram without axes or color bar
  librosa.display.specshow(mel_spectrogram_db, x_axis=None, y_axis=None)

  # Remove whitespace around the image
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

  if USE_GRAYSCALE:
    plt.gray()
    plt.set_cmap("gray")

  # Save the Mel spectrogram as a PNG image
  plt.savefig(
    output_image_path,
    pil_kwargs={'compress_level': 0},
    dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True
  )

  plt.close(fig)

  if USE_GRAYSCALE:
    # Schifo perché plt non sa cos'è un PNG a singolo canale mannaggia
    i = Image.open(output_image_path)
    i.convert('L').save(output_image_path)
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

csv_f = open(output_path + "_labels.csv", "w+")

audio_tensor_list = []
for audio_file in os.listdir(dataset_path):

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

  csv_f.write(f"{emotion}/{pre}.png,{emotion}\n")

csv_f.close()

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

for actor_folder in ["m3","m2","m1","f3","f2","f1"]:
  folderpath = os.path.join(dataset_path, actor_folder)
  print("Analysis of " + folderpath)

  for audio_file in os.listdir(folderpath):
    emotionCode = audio_file[0:3]

    if emotionCode not in emovo_emotions_codes:
      print("Unknown emotion " + emotionCode)
      continue

    pre, ext = os.path.splitext(audio_file)
    out_path =  os.path.join(output_path, emovo_emotions_codes[emotionCode], f"emovo_{pre}.png")

    audio_path = os.path.join(folderpath, audio_file)

    audio_to_mel_spectrogram(
        input_audio_path = audio_path,
        output_image_path = out_path,
    )

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

count = 0

for folder_name, emotion in emovdb_emotions_codes.items():
  subfolder_name = os.path.join(dataset_path, folder_name)

  for audiofilename in os.listdir(subfolder_name):
    pre, ext = os.path.splitext(audiofilename)

    # Generate and save to disk
    audio_path = os.path.join(subfolder_name, audiofilename)
    output_filename = os.path.join(output_path, f"{emotion}/{pre}.png")

    audio_to_mel_spectrogram(
      input_audio_path = audio_path,
      output_image_path = output_filename,
    )

    count += 1
    if count % 100 == 0:
      gc.collect()

RE_EXTRACT_EMOTION = re.compile(r"\w+_(\w+)_.*_.*\.wav")

jl_emotions_codes = {
  "angry": "ANGER",
  "anxious": "ANXIETY",
  "happy": "HAPPINESS",
  "sad": "SADNESS",
  "neutral": "NEUTRAL",
  "excited": "HAPPINESS", # @TODO DECIDE IF TRUE LOL
}

dataset_path = os.path.join(root, "jl-corpus/")

for audio_file in os.listdir(dataset_path):
  pre, ext = os.path.splitext(audio_file)

  # in this dataset the emotionCode is written inside the name
  emotionCode = RE_EXTRACT_EMOTION.findall(audio_file)[0]
  if emotionCode not in jl_emotions_codes:
    print("Unknown emotion " + emotionCode)
    continue

  out_path =  os.path.join(output_path, jl_emotions_codes[emotionCode], f"jl_{pre}.png")

  audio_path = os.path.join(dataset_path, audio_file)

  audio_to_mel_spectrogram(
      input_audio_path = audio_path,
      output_image_path = out_path,
  )
  gc.collect()

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
