import os
from pathlib import Path

from flask import app, json
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from groq import Groq, RateLimitError
from pydub import AudioSegment
import torchaudio.transforms as T

from constans import SECOND_FILE_PATH, THIRD_FILE_PATH, FIRST_FILE_PATH, COMBINED_FILE_PATH

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL_LANGUAGE = os.environ["MODEL_LANGUAGE"]
MODELS_DICT = json.loads(os.environ["MODELS_DICT"])


class SpeechToText:
    def __init__(self):
        self.model, self.processor = self._init_whisper_model_and_processor()

    def _init_whisper_model_and_processor(self):
        model_name = MODELS_DICT[MODEL_LANGUAGE]
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        processor = WhisperProcessor.from_pretrained(model_name)
        return model, processor

    def get_file_path_for_transcribe(self) -> str:
        if not Path(SECOND_FILE_PATH).is_file():
            return THIRD_FILE_PATH

        if not Path(FIRST_FILE_PATH).is_file():
            sound1: AudioSegment = AudioSegment.from_file(SECOND_FILE_PATH, format="wav")
            sound2: AudioSegment = AudioSegment.from_file(THIRD_FILE_PATH, format="wav")
            combined: AudioSegment = sound1 + sound2
            combined.export(COMBINED_FILE_PATH, format="wav")
            return COMBINED_FILE_PATH

        sound1: AudioSegment = AudioSegment.from_file(FIRST_FILE_PATH, format="wav")
        sound2: AudioSegment = AudioSegment.from_file(SECOND_FILE_PATH, format="wav")
        sound3: AudioSegment = AudioSegment.from_file(THIRD_FILE_PATH, format="wav")
        combined: AudioSegment = sound1 + sound2 + sound3
        combined.export(COMBINED_FILE_PATH, format="wav")
        return COMBINED_FILE_PATH

    def transcribe_with_groq(self, file_path: str) -> str:
        try:
            client = Groq(api_key=GROQ_API_KEY)

            with open(file_path, "rb") as file:
                transcription = client.audio.transcriptions.create(
                    file=(file_path, file.read()),
                    model="whisper-large-v3",
                    prompt="Specify context or spelling",
                    response_format="json",
                    language=MODEL_LANGUAGE,
                    temperature=0.0
                )
                return transcription.text
        except RateLimitError:
            raise RateLimitError

    """
    Loads the audio file, converts stereo to mono if necessary, resamples to 16000 Hz if needed, 
    and processes it with the Whisper processor.
    """

    def _load_and_process_audio(self, filepath):
        audio_input, sr = torchaudio.load(filepath, format='wav')
        # Convert stereo to mono by averaging the channels if necessary
        if audio_input.shape[0] > 1:
            audio_input = torch.mean(audio_input, dim=0, keepdim=True)
        # Check and resample the audio if necessary
        if sr != 16000:
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            audio_input = resampler(audio_input.squeeze())
        audio_input = audio_input.squeeze().squeeze()
        input_features = self.processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
        return input_features

    """
    Uses the model to generate the transcription from the input features.
    """

    def _generate_transcription(self, input_features):
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

    def transcribe_with_whisper(self, file_path: str) -> str:
        input_features = self._load_and_process_audio(file_path)
        transcription = self._generate_transcription(input_features)
        return transcription


speech_to_text = SpeechToText()
