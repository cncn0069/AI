from utils import data_utils as du
import numpy as np

# yt_url = "https://www.youtube.com/watch?v=zr-s47f-DOA"
# download_dir = "./data/"
#
# audio_path = du.download_and_convert(yt_url, download_dir=download_dir)
audio_path = r"C:\Users\admin\Documents\git_clone\AI\Voice similarity\data\1\1-5.wav"
# du.wav_cutter(audio_path=audio_path, output_dir=download_dir + "1/")

norm_S, sr, mfccs_40, mfccs_80 = du.audio_to_mfcc(audio_path=audio_path)
print("norm_S: ", norm_S)
print("sr: ", sr)
print("norm_S's shape: ", np.shape(norm_S))
print("mfccs_20's shape: ", np.shape(mfccs_40))
print("mfccs_40's shape: ", np.shape(mfccs_80))
print(mfccs_40)
