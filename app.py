import os
from flask import Flask, request, jsonify
from TTS.api import TTS
from pydub import AudioSegment

# Initialize Flask app
app = Flask(__name__)

# Load the TTS model (you can keep it generic for multilingual use)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

def synthesize_in_chunks(text, file_path="output.wav", speaker="Ana Florence", language="hi", chunk_size=200):
    text_chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    temp_files = []

    for i, chunk in enumerate(text_chunks):
        temp_file = f"temp_chunk_{i}.wav"
        tts.tts_to_file(text=chunk, file_path=temp_file, speaker=speaker, language=language)
        temp_files.append(temp_file)

    combined_audio = AudioSegment.empty()
    for temp_file in temp_files:
        combined_audio += AudioSegment.from_file(temp_file)
        os.remove(temp_file)

    combined_audio.export(file_path, format="wav")
    return file_path

# API endpoint to generate speech
@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text', '')
    speaker = data.get('speaker', 'Ana Florence')
    language = data.get('language', 'hi')  # Default to Hindi if no language is specified

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Generate the audio
    output_file = "output.wav"
    synthesize_in_chunks(text, file_path=output_file, speaker=speaker, language=language)

    return jsonify({"message": f"Audio generated successfully: {output_file}"}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
