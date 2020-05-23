from utils import data_utils as du


yt_url = "https://www.youtube.com/watch?v=zr-s47f-DOA"
download_dir = "./data/"

# audio_path = du.download_and_convert(yt_url, download_dir=download_dir)
audio_path = r"C:\Users\admin\Documents\git_clone\AI\Voice similarity\data\1.wav"
du.wav_cutter(audio_path=audio_path, output_dir=download_dir + "1/")
