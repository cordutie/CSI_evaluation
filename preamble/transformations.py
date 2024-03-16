import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from IPython.display import display, Audio
import essentia.standard as es
from essentia.pytools.spectral import hpcpgram

sys.path.append('..')
import libfmp.b
import libfmp.c4
import libfmp.c7

def load_audio(path, Fs):
    y_og, fs_og, n_channels, _, _, _ = es.AudioLoader(filename=path)()
    y_mono       = es.MonoMixer()(y_og, 2)
    y_mono_fs = es.Resample(inputSampleRate=fs_og, outputSampleRate=Fs, quality=1).compute(y_mono)
    return y_mono_fs

def change_tempo(input_folder, output_folder, tempo_change_factor, label):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for filename in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, filename)
    y, sr = librosa.load(input_file_path)
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_change_factor)

    output_file_path = os.path.join(output_folder, label + '_' + filename)
    sf.write(output_file_path, y_stretched, sr)

    print(f"Tempo changed for {filename} and saved as {output_file_path}")


def change_pitch(input_file_path, output_file_path, pitch_steps, label):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for filename in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, filename)
    y, sr = librosa.load(input_file_path, sr=None)
    y_pitch_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_steps)

    output_file_path = os.path.join(output_folder, label + '_' + filename)
    sf.write(output_file_path, y_pitch_shifted, sr)

    print(f"Pitch changed for {filename} and saved as {output_file_path}")