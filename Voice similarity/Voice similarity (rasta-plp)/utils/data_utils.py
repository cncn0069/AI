from sidekit.features_extractor import plp
from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import librosa.display
import librosa
import numpy as np
import os


def file_name_change(dir_path):
    file_list = os.listdir(dir_path)
    for i in range(len(file_list)):
        org_name = dir_path + "\\" + file_list[i]
        chg_name = dir_path + "\\" + str(i) + ".wav"
        os.rename(src=org_name, dst=chg_name)


def raw_data_processing(data_path):
    data = list()
    label = list()

    file_dir = os.listdir(path=data_path)

    for i in range(len(file_dir)):
        file_list = os.listdir(data_path + "\\" + file_dir[i])
        for j in range(len(file_list)):
            file_name = data_path + "\\" + file_dir[i] + "\\" + file_list[j]
            print("file_name: ", file_name)
            audio_file, sr = librosa.load(path=file_name, sr=int(mediainfo(file_name)['sample_rate']))

            try:
                audio_plp = plp(input_sig=audio_file, fs=sr)
                data.append(audio_plp[0])
                label.append(i)
            except:
                print("Error occured at ", file_name)
                continue

    data = np.asarray(data)
    label = np.asarray(label)

    return data, label
