import logging
import os
import tarfile
import time
import traceback
import urllib.request
import uuid
from threading import Timer

import sherpa_onnx
import soundfile as sf
from flask import Blueprint, jsonify, request, send_from_directory

# Initialize Flask Blueprint
speech_bp = Blueprint("speech", __name__)

# Constants for model URLs
MODEL_URLS = {
    "vits-piper-fa-haaniye_low": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fa-haaniye_low.tar.bz2",
    "vits-mimic3-fa-haaniye_low": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-mimic3-fa-haaniye_low.tar.bz2",
    "vits-piper-fa_en-rezahedayatfar-ibrahimwalk-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fa_en-rezahedayatfar-ibrahimwalk-medium.tar.bz2",
    "matcha-tts-fa_en-musa": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-tts-fa_en-musa.tar.bz2",
    "matcha-tts-fa_en-khadijah": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-tts-fa_en-khadijah.tar.bz2",
    "vits-piper-fa_IR-gyro-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fa_IR-gyro-medium.tar.bz2",
    "vits-piper-fa_IR-ganji_adabi-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fa_IR-ganji_adabi-medium.tar.bz2",
    "vits-piper-fa_IR-reza_ibrahim-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fa_IR-reza_ibrahim-medium.tar.bz2",
    "vits-piper-fa_IR-ganji-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fa_IR-ganji-medium.tar.bz2",
    "vits-piper-fa_IR-amir-medium": "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-fa_IR-amir-medium.tar.bz2",
}

MODEL_DIR = "/home/dev/.cache/tts-models"
GENERATED_FILES_DIR = "/home/dev/.cache/generated_audios"

# Default configurations that will be used for each model
DEFAULT_MODEL_CONFIGS = {
    "vits-piper-fa_IR-amir-medium": {
        "speed": 1.0,  # Default speed
        "provider": "cuda",
        "debug": False,
        "num_threads": 1,
    },
    "vits-piper-fa-haaniye_low": {
        "speed": 1.0,  # Default speed
        "provider": "cuda",
        "debug": False,
        "num_threads": 1,
    },
}


# Helper function to download and extract a model
def download_and_extract_model(model_name: str):
    model_url = MODEL_URLS.get(model_name)
    if not model_url:
        raise ValueError(f"Model '{model_name}' not found in available models.")

    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        logging.info(f"Model '{model_name}' not found, downloading...")

        # Download model tar.bz2 file
        tar_file_path = f"{model_name}.tar.bz2"
        urllib.request.urlretrieve(model_url, tar_file_path)

        # Extract the model tar.bz2 file
        with tarfile.open(tar_file_path, "r:bz2") as tar:
            tar.extractall(path=MODEL_DIR)

        os.remove(tar_file_path)
        logging.info(f"Model '{model_name}' extracted to {MODEL_DIR}")
    else:
        logging.info(f"Model '{model_name}' already exists in {MODEL_DIR}")


# Function to generate speech
def generate_speech(text: str, model_name: str, speed: float):
    os.makedirs(GENERATED_FILES_DIR, exist_ok=True)

    text = text.replace("**", "")

    # Generate file path
    file_name = f"{model_name}_{uuid.uuid4().hex[:8]}.wav"
    file_path = os.path.join(GENERATED_FILES_DIR, file_name)

    try:
        # Construct the model paths based on model name
        model_dir = os.path.join(MODEL_DIR, model_name)
        model_file = os.path.join(
            model_dir, f"{model_name.replace('vits-piper-', '')}.onnx"
        )
        lexicon = ""
        tokens = os.path.join(model_dir, "tokens.txt")
        data_dir = os.path.join(model_dir, "espeak-ng-data")
        dict_dir = ""

        # Get default configuration for the model (or use passed speed)
        model_config = DEFAULT_MODEL_CONFIGS.get(model_name, {})
        speed = model_config.get("speed", speed)
        provider = model_config.get("provider", "cpu")
        debug = model_config.get("debug", False)
        num_threads = model_config.get("num_threads", 1)
        rule_fsts = ""
        max_num_sentences = -1

        # Create TTS config with model-specific settings
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=model_file,
                    lexicon=lexicon,
                    data_dir=data_dir,
                    dict_dir=dict_dir,
                    tokens=tokens,
                ),
                provider=provider,
                debug=debug,
                num_threads=num_threads,
            ),
            rule_fsts=rule_fsts,
            max_num_sentences=max_num_sentences,
        )

        print(tts_config)

        # if not tts_config.validate():
        #     raise ValueError("Invalid TTS configuration")

        tts = sherpa_onnx.OfflineTts(tts_config)

        # Start speech generation
        start = time.time()
        audio = tts.generate(text, sid=0, speed=speed)
        end = time.time()

        if len(audio.samples) == 0:
            return jsonify({"error": "Error generating audio."}), 500

        elapsed_seconds = end - start
        audio_duration = len(audio.samples) / audio.sample_rate
        real_time_factor = elapsed_seconds / audio_duration

        # Save the generated audio file
        sf.write(
            file_path, audio.samples, samplerate=audio.sample_rate, subtype="PCM_16"
        )

        # Schedule file deletion
        schedule_file_deletion(file_path)

        return jsonify({"file_url": f"{file_path}"}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Function to schedule file deletion
def schedule_file_deletion(file_path, delay=600):
    def delete_file():
        if os.path.exists(file_path):
            os.remove(file_path)

    Timer(delay, delete_file).start()


# Flask route to generate speech
@speech_bp.route("/generate-speech", methods=["POST"])
def generate_speech_route():
    try:
        data = request.json
        text = data.get("text")
        model_name = data.get("model", "vits-piper-fa_IR-amir-medium")  # Default model
        speed = data.get("speed", 1.0)  # Speed parameter

        if not text:
            return jsonify({"error": "Missing text input."}), 400

        # Download and extract model if not already present
        download_and_extract_model(model_name)
        return generate_speech(text, model_name, speed)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# Flask route to serve generated audio
@speech_bp.route("/download-audio/<filename>", methods=["GET"])
def download_audio(filename):
    return send_from_directory(GENERATED_FILES_DIR, filename, as_attachment=True)
