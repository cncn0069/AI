from pydub.utils import mediainfo
import librosa
import pytube
import pydub


def download_and_convert(url, download_dir):
    yt = pytube.YouTube(url)
    yt_audio = yt.streams.filter(type="audio").first()
    download_path = yt_audio.download(output_path=download_dir, filename="1")
    rename_path = download_path[:-4] + ".wav"

    tmp = pydub.AudioSegment.from_file(file=download_path, format="mp4")
    tmp.export(out_f=rename_path, format="wav")
    # os.remove(download_path)

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
