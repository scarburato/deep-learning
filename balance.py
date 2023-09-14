import os
import random

path = "build/processed/NEUTRAL"

for file in os.listdir(path):
  if random.random() < (1.0 - 1.0/5.0):
    os.remove(os.path.join(path, file))

path = "build/processed/HAPPINESS"

for file in os.listdir(path):
  if random.random() < (1.0 - 0.425):
    os.remove(os.path.join(path, file))

path = "build/processed/ANGER"

for file in os.listdir(path):
  if random.random() < (1.0 - 0.52):
    os.remove(os.path.join(path, file))