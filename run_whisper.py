import gemini_helper
from flask import Flask, request, jsonify
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
import torch
import torchaudio
import torchaudio.transforms as T
import os
import re

app = Flask(__name__)
MODEL_LANGUAGE = os.environ["MODEL_LANGUAGE"]
UPLOAD_FOLDER = os.environ["UPLOAD_FOLDER"]
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
SUPPORTED_COMMANDS = ["play", "pause", "stop", "next", "previous", "mute", "unmute", "volume up",
                      "volume down", "fullscreen", "exit fullscreen", "loop", "exit"]


def init_model_and_processor():
    models_dict = {"en": "openai/whisper-small.en", "he": "ivrit-ai/whisper-large-v2-tuned"}
    model_name = models_dict[MODEL_LANGUAGE]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = WhisperProcessor.from_pretrained(model_name)
    return model, processor


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


"""
Saves the uploaded file to the specified upload folder and returns the file path
@param file: A file to save.
"""


def save_file(file: FileStorage) -> str:
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    return filepath


def del_file(filename: str) -> bool:
    try:
        # Construct the filepath using the secure filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))

        # Check if the file exists
        if os.path.exists(filepath):
            # Remove the file
            os.remove(filepath)
            return True
        else:
            print("File not found.")
            return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False
"""
Loads the audio file, converts stereo to mono if necessary, resamples to 16000 Hz if needed, 
and processes it with the Whisper processor.
"""


def load_and_process_audio(processor, filepath):
    audio_input, sr = torchaudio.load(filepath, format='wav')
    # Convert stereo to mono by averaging the channels if necessary
    if audio_input.shape[0] > 1:
        audio_input = torch.mean(audio_input, dim=0, keepdim=True)
    # Check and resample the audio if necessary
    if sr != 16000:
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        audio_input = resampler(audio_input.squeeze())
    audio_input = audio_input.squeeze().squeeze()
    input_features = processor(audio_input, sampling_rate=16000, return_tensors="pt").input_features
    return input_features


"""
Uses the model to generate the transcription from the input features.
"""


def generate_transcription(processor, model, input_features):
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription


"""
Removes punctuation from the transcription.
"""


def sanitize_transcription(transcription):
    return re.sub(r'[^\w\s]', '', transcription)


def find_commands(sanitized_transcription):
    found_commands = [command for command in SUPPORTED_COMMANDS if command in sanitized_transcription.lower()]
    return found_commands


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    model = app.config['MODEL']
    processor = app.config['PROCESSOR']
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filepath = save_file(file)
        input_features = load_and_process_audio(processor, filepath)
        transcription = generate_transcription(processor, model, input_features)
        sanitized_transcription = sanitize_transcription(transcription)
        found_commands = find_commands(sanitized_transcription)
        if not found_commands:
            gemini_recognized_action = gemini_helper.check_for_video_action(sanitized_transcription)
            print(gemini_recognized_action)
            if gemini_recognized_action != "false":
                found_commands.append(gemini_recognized_action)
        print(transcription)
        del_file(file)
        return jsonify({'transcription': transcription, 'commands': found_commands})
    else:
        return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    app_model, app_processor = init_model_and_processor()
    app.config['MODEL'] = app_model
    app.config['PROCESSOR'] = app_processor
    app.run(debug=True, port=4000, host='0.0.0.0')
