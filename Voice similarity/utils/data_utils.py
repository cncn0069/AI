from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import librosa
import pytube
import pydub
import time


def download_and_convert(url, download_dir):
    yt = pytube.YouTube(url)
    yt_audio = yt.streams.filter(type="audio").first()
    download_path = yt_audio.download(output_path=download_dir, filename="1")
    rename_path = download_path[:-4] + ".wav"

    tmp = pydub.AudioSegment.from_file(file=download_path, format="mp4")
    tmp.export(out_f=rename_path, format="wav")

    return rename_path


def wav_cutter(audio_path, output_dir):
    file_path_m = int(float(mediainfo(audio_path)['duration']) / 60)

    y, sr = librosa.load(path=audio_path, sr=int(mediainfo(audio_path)['sample_rate']))
    print("len(y): ", len(y), "\nsr: ", sr)

    cut = len(y) / file_path_m
    eof = len(y)
    print("cut: ", cut, "eof: ", eof)

    t1, t2 = 0, int(cut)
    print("t2: ", t2)
    index = 0

    y_original = y
    while t2 <= eof:
        print(index)
        y = y_original[t1:t2]
        librosa.output.write_wav((output_dir + "1-" + str(index) + ".wav"), y, sr)
        index += 1
        t1 = t2
        t2 = t2 + int(cut)


def audio_to_mfcc(audio_path):
    y, sr = librosa.load(path=audio_path, sr=int(mediainfo(audio_path)['sample_rate']))

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    log_S = librosa.amplitude_to_db(S=S)

    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.show()
    time.sleep(0.3)

    min_level_db = -100

    def _normalize(S):
        return np.clip((S - min_level_db) / -min_level_db, 0, 1)

    norm_S = _normalize(log_S)

    librosa.display.specshow(norm_S, sr=sr, x_axis='time', y_axis='mel')
    plt.title('norm mel power spectrogram')
    plt.colorbar(format='%+0.1f dB')
    plt.show()
    time.sleep(0.3)

    mfccs_40 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_80 = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=80)

    librosa.display.specshow(mfccs_40, x_axis="time")
    plt.title("MFCC with n_mfcc=40")
    plt.show()
    time.sleep(0.3)

    librosa.display.specshow(mfccs_80, x_axis="time")
    plt.title("MFCC with n_mfcc=80")
    plt.show()
    time.sleep(0.3)

    return norm_S, sr, mfccs_40, mfccs_80
