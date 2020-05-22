import subprocess
import pytube
import pydub
from pydub.utils import mediainfo
import os


def download_and_convert(url, download_dir):
    yt = pytube.YouTube(url)
    yt_audio = yt.streams.filter(type="audio").first()
    download_path = yt_audio.download(output_path=download_dir, filename="1")
    rename_path = download_path[:-4] + ".wav"

    original_bitrate = mediainfo(download_path)['bit_rate']

    tmp = pydub.AudioSegment.from_file(file=download_path, format="mp4")
    tmp.export(out_f=rename_path, format="wav")
    # os.remove(download_path)

    return rename_path
