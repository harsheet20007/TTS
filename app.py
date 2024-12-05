from flask import Flask, request, jsonify
from TTS.api import TTS
import os
from pydub import AudioSegment

app = Flask(__name__)

# Initialize the TTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text')
    speaker =  data.get('Ana Florence')
    language = data.get('language', 'en')  # Default to English if no language is provided
    output_file = f"{language}_output.wav"

    # Process text-to-speech
    tts.tts_to_file(text=text, file_path=output_file, language=language, speaker=speaker)

    # Return the audio file URL (if hosted) or a success message
    return jsonify({"message": "Audio generated successfully", "file": output_file})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
