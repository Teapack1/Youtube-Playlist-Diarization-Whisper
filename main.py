"""
Install:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
"""

from pytube import Playlist, YouTube, Channel
from pydub import AudioSegment
import torch
import os
from diarize_class import DiarizePipeline

# torch.backends.cuda.preferred_linalg_library("default")

print(torch.version.cuda)

device = "cuda"
batch_size = 8  # reduce if low on GPU mem
mtypes = {"cpu": "int8", "cuda": "float32"}
DATASET_PATH = "Downloads/dataset"
MODEL_PATH = "Downloads/model"
DOWNLOADS = "Downloads"
WORKING_AUDIO = os.path.join(DOWNLOADS, "processed_audio_file.mp3")
PLAYLIST_URL = (
    "https://www.youtube.com/playlist?list=PLkL7BvJXiqSQu3i72hSrG4vUkDuaneHuB"
)

DOWNLOAD_START = 1  # Start from video number
DOWNLOAD_NUMBER = 999  # Max number of videos to download
CHANNEL_URL = "https://www.youtube.com/channel/UCIaH-gZIVC432YRjNVvnyCA/shorts"
shorts = False

# Lex Fridman's podcast https://www.youtube.com/watch?v=zMYvGf7BA9o&list=PLrAXtmErZgOdP_8GztsuKi9nrraNbKKp4
# Test Playlist shorts https://www.youtube.com/playlist?list=PLPWDuIHYjJBHFaOlKGrn2aHy4AHarss6A
# Williamson podcast https://www.youtube.com/playlist?list=PLkL7BvJXiqSQu3i72hSrG4vUkDuaneHuB
# Williamson shorts https://www.youtube.com/channel/UCIaH-gZIVC432YRjNVvnyCA/shorts

if not os.path.exists(DOWNLOADS):
    os.mkdir(DOWNLOADS)
    os.mkdir(MODEL_PATH)
    os.mkdir(DATASET_PATH)

DiarizePipeline = DiarizePipeline(result_file_path=DATASET_PATH, mtypes=mtypes)


if shorts:
    print("\nLoading all shorts from the channel, this might take a minute or two...")
    playlist = Channel(CHANNEL_URL).shorts
    playlist = playlist[DOWNLOAD_START - 1 : len(playlist)]

else:
    print("\nLoading all videos from the playlist, this might take a minute or two...")
    playlist = Playlist(PLAYLIST_URL).videos

print(f"Number of videos in playlist: {len(playlist)}")

for number, vid in enumerate(playlist, start=DOWNLOAD_START - 1):
    print(
        f"Processing video: {number+1} of {DOWNLOAD_NUMBER} (len: {len(playlist)})..."
    )

    if number == DOWNLOAD_NUMBER:
        print(f"{number+1} of video downloaded. Exitting...")
        break

    # Clear existing mp4 files
    clear_mp4s = [file for file in os.listdir(DOWNLOADS) if file.endswith(".mp4")]
    for mp4_file in clear_mp4s:
        os.remove(os.path.join(DOWNLOADS, mp4_file))

    title = vid.title
    print(f"Downloading:\n{title}")
    print(vid.streams.filter(only_audio=True))

    target_stream = (
        vid.streams.filter(only_audio=True, file_extension="mp4", abr="128kbps")
        .order_by("abr")
        .desc()
    )
    print(target_stream)
    target_stream.first().download(DOWNLOADS)
    print(f"Downloaded: {target_stream.first()}")

    print("Converting from mp4 to mp3...")

    downloaded_audio_file = [
        file for file in os.listdir(DOWNLOADS) if file.endswith(".mp4")
    ]
    downloaded_audio_file = os.path.join(DOWNLOADS, downloaded_audio_file[0])
    print(downloaded_audio_file)
    audio_file = AudioSegment.from_file(downloaded_audio_file, format="mp4")
    audio_file.export(WORKING_AUDIO, format="mp3")

    print("Deleting mp4 file...")
    os.remove(downloaded_audio_file)

    print("Processing diarization pipeline...")

    DiarizePipeline(
        audio=WORKING_AUDIO,  # "name of the target audio file"
        stemming=False,  # "Disables source separation. This helps with long files that don't contain a lot of music."
        suppress_numerals=True,  # "Suppresses Numerical Digits. This helps the diarization accuracy but converts all digits into written text.
        model_name="medium.en",  # "name of the Whisper model to use"
        batch_size=batch_size,  # "Batch size for batched inference, reduce if you run out of memory, set to 0 for non-batched inference"
        language=None,  # "Language spoken in the audio, specify None to perform language detection",
        device=device,  # "if you have a GPU use 'cuda', otherwise 'cpu'",
        model_path=MODEL_PATH,  # "path to the folder where the model will be downloaded to"
        title=title,  # "title of the video"
    )
