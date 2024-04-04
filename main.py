"""
Install:
Python 3.10
"https://visualstudio.microsoft.com/cs/visual-cpp-build-tools/

pip install Cython
pip install youtokentome
pip install -r requirements.txt
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 --timeout 300 --retries 100
"""

import argparse
import time
import yt_dlp
import os
from diarize_class import DiarizePipeline
from helpers import split_audio

args = argparse.ArgumentParser()
args.add_argument(
    "-url",
    help="url of the playlist",
    required=True,
    type=str,
    default="https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4",  # Add your playlist URL
)
args.add_argument(
    "-min",
    help="playlist start video no.",
    required=False,
    type=int,
    default=1,
)
args.add_argument(
    "-max",
    help="playlist end video no.",
    required=False,
    type=int,
    default=999,
)
args.add_argument(
    "-batch",
    help="batch size",
    required=False,
    type=int,
    default=4,
)
args.add_argument(
    "-device",
    help="device used for diarization",
    required=False,
    type=str,
    default="cuda",
)
args.add_argument(
    "-mtypes",
    help="precission settings, use dictionary format i.e.: {'cpu': 'int8', 'cuda': 'int8_float16'}",
    required=False,
    type=dict,
    default={"cpu": "int8", "cuda": "float16"},
)
args.add_argument(
    "-episodes",
    help="How many episodes to download and put into one output file (default: 100)",
    required=False,
    type=int,
    default=100,
)

args = args.parse_args()

device = args.device  # "cuda" or "cpu"
batch_size = args.batch  # reduce if low on GPU mem
mtypes = args.mtypes
DATASET_PATH = "Downloads/dataset"
MODEL_PATH = "Downloads/model"
DOWNLOADS = "Downloads"
WORKING_AUDIO = os.path.join(DOWNLOADS, "processed_audio_file")
DOWNLOAD_URL = args.url
DOWNLOAD_START = args.min  # Start from video number
DOWNLOAD_NUMBER = args.max  # Max number of videos to download
EPISODES = args.episodes  # How many episodes to download and put into one output file
nth_output_file = 0  #  n.th number of output file where output will be stored

retry_delay = 30  # Delay between retries in seconds
max_retries = 5

# Lex Fridman's podcast https://www.youtube.com/playlist?list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4
# Williamson podcast https://www.youtube.com/playlist?list=PLkL7BvJXiqSQu3i72hSrG4vUkDuaneHuB
# Williamson shorts https://www.youtube.com/channel/UCIaH-gZIVC432YRjNVvnyCA/shorts


if not os.path.exists(DOWNLOADS):
    os.makedirs(DOWNLOADS, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(DATASET_PATH, exist_ok=True)


DiarizePipeline = DiarizePipeline(result_file_path=DATASET_PATH, mtypes=mtypes)


ydl_opts_download = {
    "format": "bestaudio/best",
    "outtmpl": WORKING_AUDIO,
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }
    ],
    "prefer_ffmpeg": True,
    "keepvideo": False,
    "quiet": False,  # Path to the cookies file
    "ignoreerrors": True,  # Continue on download errors
    "socket_timeout": retry_delay,  # Timeout for network operations in seconds
    "retries": max_retries,  # Number of retries for a failed download
}

ydl_opts_info = {
    "quiet": False,
    "extract_flat": True,  # Only extract information about the entries in the playlist
    "force_generic_extractor": True,  # Force using the generic extractor
}


best_attempt_info = None
max_urls_count = 0
best_attempt_time = None


for attempt in range(1):
    try:
        with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
            info_dict = ydl.extract_info(DOWNLOAD_URL, download=False)

            video_info = [
                {
                    "title": entry["title"],
                    "url": entry["url"],
                    "duration": entry["duration"],
                }
                for entry in info_dict["entries"]
                if entry.get("url") and entry.get("title")
            ]
        if len(video_info) > max_urls_count:
            max_urls_count = len(video_info)
            best_attempt_info = video_info
            best_attempt_time = attempt

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")
        if attempt < max_retries - 1:
            print(f"[INFO] Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        else:
            print("[ERROR] Maximum retries reached, skipping video.")
            break

video_info = best_attempt_info

print(
    f"    [INFO] Best attempt ({best_attempt_time+1}) \033[1mfetched \033[4m{len(video_info)} videos\033[0m from the playlist"
)

video_info = video_info[DOWNLOAD_START - 1 : DOWNLOAD_NUMBER]
print(
    f"    [INFO] Downloading videos no.: ({DOWNLOAD_START-1} - {DOWNLOAD_NUMBER}) from the playlist"
)


for number, vid in enumerate(video_info):

    title = vid["title"]
    url = vid["url"]
    duration = vid["duration"]

    if number == DOWNLOAD_NUMBER:
        print(
            f"[INFO] Reached the final number ({number+1}) of videos downloaded. Exitting..."
        )
        break
    print(
        f"\n    [INFO] \033[1mProcessing video:\033[0m \033[4m'{title}'\033[0m: {number+1} of {DOWNLOAD_NUMBER} (len: {len(video_info)})...\n       URL: {url}\n"
    )
    if title == "[Private video]" or title == "[Deleted video]":
        print(f"[INFO] Skipping video: {title}")
        continue

    if duration:
        if duration > 10800:  # Longer than 3 hours
            ydl_opts_download["postprocessors"][0]["preferredquality"] = "128"
            print(f"[INFO] Video is longer than 3 hours, setting quality 128kbps...")
        else:
            ydl_opts_download["postprocessors"][0]["preferredquality"] = "192"
    else:
        ydl_opts_download["postprocessors"][0]["preferredquality"] = "192"

    # Setup output file for diarization text to store
    if number % EPISODES == 0:
        nth_output_file += 1
        print(f"[INFO] {nth_output_file}. output file created")

    # Attempt to download the video(audio) from the current URL:
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts_download) as ydl_audio:
                ydl_audio.download([url])

            # Did audio file downloaded successfully? If not raise an exception:
            if not os.path.exists(f"{WORKING_AUDIO}.mp3"):
                raise FileNotFoundError(f"[ERROR] No audio file found for: {title}")

            print(f"[INFO] Downloaded: {title}")
            print("[INFO] Processing diarization pipeline...")

            # Split audio and diarize:
            audio_file_path = f"{WORKING_AUDIO}.mp3"

            DiarizePipeline(
                audio=audio_file_path,  # "Diarization target audio file"
                stemming=False,  # "Disables source separation. This helps with long files that don't contain a lot of music."
                suppress_numerals=True,  # "Suppresses Numerical Digits. This helps the diarization accuracy but converts all digits into written text.
                model_name="medium.en",  # "name of the Whisper model to use"
                batch_size=batch_size,  # "Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference"
                language=None,  # "Language spoken in the audio, specify None to perform language detection",
                device=device,  # "if you have a GPU use 'cuda', otherwise 'cpu'",
                model_path=MODEL_PATH,  # "path to the folder where the model will be downloaded to"
                title=title,  # "title of the video"
                nth_output_file=nth_output_file,  # "n.th number of output file where output will be stored"
            )

            os.remove(audio_file_path)
            print(f"[INFO] Deleted audio file: {WORKING_AUDIO}.mp3")
            break  # Exit retry loop after successful download and processing

        except Exception as e:
            print(f"[ERROR] An error occurred: {e}")
            if attempt < max_retries - 1:
                print(f"[INFO] Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("[ERROR] Maximum retries reached, skipping video.")
                break  # Skip to the next video
