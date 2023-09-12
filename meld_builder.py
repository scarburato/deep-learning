import csv
import os
import ffmpeg
import sys
from datetime import datetime, timedelta
from multiprocessing import Pool

MIN_LEN = 1.5


def from_timestamp_to_seconds(timestamp: str):
  time_obj = datetime.strptime(timestamp, "%H:%M:%S,%f")
  seconds = (time_obj - datetime(1900, 1, 1)).total_seconds()

  return seconds

def process_tuple(datum):
  duration = from_timestamp_to_seconds(datum["EndTime"]) - from_timestamp_to_seconds(datum["StartTime"])
  if duration < MIN_LEN:
    return

  inputfilename = f"dia{datum['Dialogue_ID']}_utt{datum['Utterance_ID']}.mp4"
  inputfilepath = os.path.join(input_root, inputfilename)

  outputfilename = f"{datum['Sr No.']}_{datum['Emotion']}_tr.ogg"
  outputfilepath = os.path.join(output_root, outputfilename)

  stream = ffmpeg.input(inputfilepath, vn=None)
  stream = ffmpeg.output(stream, outputfilepath, format='ogg', acodec='libvorbis')

  stream.run()

csvfile = open(sys.argv[1])
csvreader = csv.DictReader(csvfile)

output_root = sys.argv[2]
if not os.path.exists(output_root):
  os.mkdir(output_root)

input_root = sys.argv[3]

assert os.path.isdir(output_root) and os.path.isdir(input_root)

with Pool(17) as p:
  p.map(process_tuple, csvreader)