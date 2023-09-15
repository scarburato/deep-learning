import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
import cv2
from math import *

WIDTH, HEIGHT = 128, 65
SAMPLES = 8000
OUT_PATH = "build/processed.synt"

os.makedirs(OUT_PATH, exist_ok=True)

classes = ["SURPRISE", "SADNESS", "NEUTRAL", "HAPPINESS", "DISGUST", "ANGER"]
functions = [
  lambda x, y: max(3/7, 0.5*sin(cos(tan(x))) + sin(cos(tan(y))) + 0.5),
  lambda x, y: 0.5*sin(cos(tan(x*y))) + 0.5,
  lambda x, y: (1/((abs(x-y)) + 1)),
  lambda x, y: 1 - exp(-sqrt(x + y)),
  lambda x, y: max(1 - exp(-sqrt(x)), 0.5*cos(y) + 0.5),
  lambda x, y: (1/((x/y) + 1))
]

for c in classes:
  os.makedirs(os.path.join(OUT_PATH, c), exist_ok=True)

rng = np.random.default_rng(0xcafebabe)

i = 0
for _ in range(0, ceil(SAMPLES/len(classes))):
  i += 1
  c = classes[i % len(classes)]
  f = functions[i % len(classes)]

  M = rng.integers(low=0, high=64, size=(HEIGHT, WIDTH), dtype=np.uint8)

  for x in range(0, len(M)):
    t = M[x]
    for y in range(0, len(t)):
      M[x][y] += (255-64-1)*f((x%9)+1,(y%13)+1)


  im = Image.fromarray(M, mode="L")
  im.save(os.path.join(OUT_PATH, c, f"{i}_{c}.png"))
  im.close()