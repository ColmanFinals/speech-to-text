from flask import Flask, request, jsonify
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from werkzeug.utils import secure_filename
import torch
import torchaudio
import torchaudio.transforms as T
import os
import re

app = Flask(__name__)
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small.en")


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Load and process the audio file
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

        # Generate token ids and decode to text
        with torch.no_grad():
            predicted_ids = model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Sanitize transcription to remove punctuation
        sanitized_transcription = re.sub(r'[^\w\s]', '', transcription)

        # Check for command words
        commands = ["play", "pause", "stop", "next", "previous", "mute", "unmute", "volume up",
                    "volume down", "fullscreen", "exit fullscreen", "loop", "exit"]
        found_commands = [command for command in commands if command in sanitized_transcription.lower()]

        return jsonify({'transcription': transcription, 'commands': found_commands})
    else:
        return jsonify({'error': 'File type not allowed'}), 400


if __name__ == '__main__':
    app.run(debug=True, port=4000, host='0.0.0.0')
