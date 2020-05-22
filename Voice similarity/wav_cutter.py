from pydub.utils import mediainfo
import matplotlib.pyplot as plt
import numpy as np
import librosa

audio_path = "./data/1.wav"
file_path_m = float(mediainfo(audio_path)['duration']) / 60

y, sr = librosa.load(path=audio_path, sr=int(mediainfo(audio_path)['sample_rate']))
print("y: ", y, "\nsr: ", sr)

# plot
cut = len(y) / int(file_path_m)
print("round(cut): ", round(cut), "\ncut: ", cut)
y = y[:round(cut)]
# time = np.linspace(0, len(y) / sr, len(y))
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.plot(time, y)
# plt.show()
librosa.output.write_wav('cut.wav', y, sr)
