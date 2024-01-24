import os
import logging
import re
import torch
from pydub import AudioSegment
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
from helpers import *
from faster_whisper import WhisperModel
import whisperx
import csv


class DiarizePipeline:
    def __init__(self, result_file_path):
        self.mtypes = {"cpu": "int8", "cuda": "float32"}
        self.result_file_path = result_file_path
        self.last_speaker = None
        self.last_segment = None

        with open(
            f"{self.result_file_path}/dataset.csv",
            mode="w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=["speaker", "text", "title", "start_time", "end_time"]
            )
            writer.writeheader()

    def __call__(
        self,
        audio,
        stemming=True,
        suppress_numerals=False,
        model_name="medium.en",
        batch_size=8,
        language=None,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_path="MODEL",
        title="",
    ):
        if stemming:
            return_code = os.system(
                f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio}" -o "temp_outputs"'
            )
            if return_code != 0:
                logging.warning(
                    "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
                )
                vocal_target = audio
            else:
                vocal_target = os.path.join(
                    "temp_outputs",
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = audio

        if batch_size != 0:
            from transcription_helpers import transcribe_batched

            whisper_results, language = transcribe_batched(
                vocal_target,
                language,
                batch_size,
                model_name,
                self.mtypes[device],
                suppress_numerals,
                device,
                model_path,
            )
        else:
            from transcription_helpers import transcribe

            whisper_results, language = transcribe(
                vocal_target,
                language,
                model_name,
                self.mtypes[device],
                suppress_numerals,
                device,
                model_path,
            )

        if language in wav2vec2_langs:
            alignment_model, metadata = whisperx.load_align_model(
                language_code=language, device=device
            )
            result_aligned = whisperx.align(
                whisper_results, alignment_model, metadata, vocal_target, device
            )
            word_timestamps = filter_missing_timestamps(
                result_aligned["word_segments"],
                initial_timestamp=whisper_results[0].get("start"),
                final_timestamp=whisper_results[-1].get("end"),
            )
            del alignment_model
            torch.cuda.empty_cache()
        else:
            assert (
                batch_size == 0
            ), "Unsupported language, use --batch_size to 0 to generate word timestamps using whisper directly."
            word_timestamps = []
            for segment in whisper_results:
                for word in segment["words"]:
                    word_timestamps.append(
                        {"word": word[2], "start": word[0], "end": word[1]}
                    )

        # Convert audio to mono for NeMo compatibility
        sound = AudioSegment.from_file(vocal_target).set_channels(1)
        ROOT = os.getcwd()
        temp_path = os.path.join(ROOT, "temp_outputs")
        os.makedirs(temp_path, exist_ok=True)
        sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")

        # Initialize NeMo MSDD diarization model
        msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(device)
        msdd_model.diarize()
        del msdd_model
        torch.cuda.empty_cache()

        # Reading timestamps <> Speaker Labels mapping
        speaker_ts = []
        with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if language in punct_model_langs:
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            words_list = list(map(lambda x: x["word"], wsm))
            labeled_words = punct_model.predict(words_list)

            ending_puncts = ".?!"
            model_puncts = ".,;:!?"
            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

            for word_dict, labeled_tuple in zip(wsm, labeled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word
        else:
            logging.warning(
                f"Punctuation restoration is not available for {language} language. Using the original punctuation."
            )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        with open(
            f"{self.result_file_path}/{self.clean_title(title)}.txt",
            "w",
            encoding="utf-8-sig",
        ) as f:
            get_speaker_aware_transcript(ssm, f)

        with open(
            f"{self.result_file_path}/{self.clean_title(title)}.srt",
            "w",
            encoding="utf-8-sig",
        ) as srt:
            write_srt(ssm, srt)

        self.append_data(ssm, title)
        cleanup(temp_path)

        ######### Write dataset #########

    def clean_title(self, title):
        replacements = {
            " ": "_",
            "|": "-",
            ":": "-",
        }
        for old, new in replacements.items():
            title = title.replace(old, new)
        return title

    def append_data(self, result_segments, title):
        with open(
            f"{self.result_file_path}/dataset.csv",
            mode="a",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=["speaker", "text", "title", "start_time", "end_time"]
            )
            print(f"SEGMENTS:\n\n {result_segments}")
            print(f"TITLE:\n\n {title}")

            first_speaker_id = None

            for segment in result_segments:
                speaker = segment.get("speaker")

                if speaker is None:
                    segment["speaker"] = self.last_speaker
                    print(
                        "Warning: 'speaker' key is missing in the segment, inserting last speaker."
                    )

                elif first_speaker_id is None:
                    first_speaker_id = segment["speaker"]

                if first_speaker_id in segment["speaker"]:
                    segment["speaker"] = "\n###Human:\n"
                else:
                    segment["speaker"] = "\n###Assistant:\n"
                print(segment)

                # Ensure the segment has all required fields
                if all(
                    key in segment
                    for key in ["speaker", "text", "start_time", "end_time"]
                ):
                    text_without_quotes = segment["text"].replace('"', "")
                    print(text_without_quotes)
                    print(self.last_segment)
                    print(self.last_speaker)
                    current_speaker = segment["speaker"]

                    # Check if the current speaker is the same as the last one
                    if self.last_speaker == current_speaker:
                        # Append text to the previous entry and update the end time
                        self.last_segment["text"] += text_without_quotes
                        self.last_segment["end_time"] = segment["end_time"]
                    else:
                        # Write the last segment if it exists
                        if self.last_segment:
                            writer.writerow(self.last_segment)

                        # Update the last speaker and last segment
                        self.last_speaker = current_speaker
                        self.last_segment = {
                            "speaker": current_speaker,
                            "text": text_without_quotes,
                            "title": title,
                            "start_time": segment["start_time"],
                            "end_time": segment["end_time"],
                        }
                else:
                    print("Missing key in segment, skipping entry...")

            # Write the last segment after the loop
            if self.last_segment:
                writer.writerow(self.last_segment)

            # Reset last speaker and segment for future calls
            self.last_speaker = None
            self.last_segment = None
