import subprocess
import pytube
import pydub
import os


def yt_download(url):
    yt = pytube.YouTube(url)
    yt_audio = yt.streams.filter(type="audio").first()
    download_path = yt_audio.download(output_path=r"C:\Users\admin\Documents\Study\School_Study\Paper\IEIE\2020"
                                                  r"-summer_paper\Code\data\woman", filename="1"
                                      )
    rename_path = download_path[:-4] + ".wav"
    # if os.path.exists(rename_path):
    #     os.remove(rename_path)
    # os.rename(download_path, rename_path)
    # download_path = rename_path
    #
    # return download_path

    print(download_path)
    print(rename_path)

    # command = "ffmpeg -i " + download_path + " -ab 160k -ac 2 -ar 44100 -vn " + rename_path
    # subprocess.call(command)

    tmp = pydub.AudioSegment.from_file(file=download_path, format="mp4")
    tmp.export(out_f=rename_path, format="wav")
