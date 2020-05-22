from utils import data_utils as du

from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np
import librosa
import os

yt_url = "https://www.youtube.com/watch?v=zr-s47f-DOA"
download_dir = "./data/"

audio_path = du.download_and_convert(yt_url, download_dir=download_dir)
