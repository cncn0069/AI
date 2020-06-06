from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import pytube
import pydub
import time
import os


def download_and_convert(url, download_dir, number):
    yt = pytube.YouTube(url)
    yt_audio = yt.streams.filter(type="audio").first()
    download_path = yt_audio.download(output_path=download_dir, filename=str(number))
    rename_path = download_path[:-4] + ".wav"

    tmp = pydub.AudioSegment.from_file(file=download_path, format="mp4")
    tmp.export(out_f=rename_path, format="wav")

    return rename_path


def wav_cutter(audio_path, output_dir):
    file_path_quat_m = int(float(mediainfo(audio_path)['duration']) / 15)

    y, sr = librosa.load(path=audio_path, sr=int(mediainfo(audio_path)['sample_rate']))

    cut = len(y) / file_path_quat_m
    eof = len(y)

    t1, t2 = 0, int(cut)
    index = 0

    y_original = y
    while t2 <= eof:
        y = y_original[t1:t2]
        librosa.output.write_wav((output_dir + str(index) + ".wav"), y, sr)
        index += 1
        t1 = t2
        t2 = t2 + int(cut)


def audio_mfcc(audio_dir):
    norm_S_list = list()
    audio_list = os.listdir(audio_dir)

    for i in range(len(audio_list)):
        audio_path = audio_dir + "\\" + audio_list[i]
        print("audio_path: ", audio_path)
        y, sr = librosa.load(path=audio_path, sr=int(mediainfo(audio_path)['sample_rate']))

        S = librosa.feature.melspectrogram(y=y, sr=sr)
        log_S = librosa.amplitude_to_db(S=S)

        min_level_db = -100

        def _normalize(S):
            return np.clip((S - min_level_db) / -min_level_db, 0, 1)

        norm_S = _normalize(log_S)
        norm_S_list.append(norm_S)

    return norm_S_list
