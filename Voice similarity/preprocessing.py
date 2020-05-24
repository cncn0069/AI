import numpy as np

from utils import data_utils as du


# yt_url = "https://www.youtube.com/watch?v=VAW9joqgzuw"      # Man
# yt_url = "https://www.youtube.com/watch?v=zr-s47f-DOA"    # Woman
# download_dir = "./data/"
#
# audio_path = du.download_and_convert(yt_url, download_dir=download_dir)
# du.wav_cutter(audio_path=audio_path, output_dir=download_dir + "1/")

label = list()
audio_dir_0 = r"C:\Users\admin\Documents\git_clone\AI\Voice similarity\data\0"
norm_S_list_0 = du.audio_mfcc(audio_dir=audio_dir_0)
for i in range(np.shape(norm_S_list_0)[0]):
    label.append(0)

audio_dir_1 = r"C:\Users\admin\Documents\git_clone\AI\Voice similarity\data\1"
norm_S_list_1 = du.audio_mfcc(audio_dir=audio_dir_1)
for i in range(np.shape(norm_S_list_1)[0]):
    label.append(1)

norm_S_list = norm_S_list_0 + norm_S_list_1
print(np.shape(norm_S_list))
print(np.shape(label))

np.savez_compressed("raw_data.npz", raw_x=norm_S_list, raw_y=label)
