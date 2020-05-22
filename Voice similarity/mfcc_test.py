import librosa
import librosa.display
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm

audio_path = "./data/1.wav"
y, sr = librosa.load(audio_path)
IPython.display.Audio(data=y, rate=sr)
