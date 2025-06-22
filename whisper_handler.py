# whisper_handler.py (Corrected with Audio "Laundering" Technique)
import runpod
import os
import base64
import tempfile
import whisper_timestamped as whisper
import json
import subprocess

# Global variable to hold the Whisper model, so we only load it once.
whisper_model = None

def cold_start():
    """
    Loads the Whisper model into memory (and VRAM).
    """
    global whisper_model
    print("Cold start: Loading Whisper model...")
    whisper_model = whisper.load_model("medium", device="cuda")
    print("Whisper model loaded successfully.")

async def handler(job):
    """
    Processes each incoming transcription request.
    """
    print(f"Received transcription job: {job['id']}")
    job_input = job['input']
    audio_base64 = job_input.get("audio_base64")

    if not audio_base64:
        return {"error": "No 'audio_base64' provided in job input."}

    global whisper_model
    if whisper_model is None:
        cold_start()

    with tempfile.TemporaryDirectory(dir="/tmp/whisper_audio") as temp_dir:
        # --- START OF CORRECTED BLOCK ---

        # 1. Define paths for the initial audio and the clean WAV output
        temp_input_audio_path = os.path.join(temp_dir, "input.audio")
        clean_wav_path = os.path.join(temp_dir, "output.wav")

        # 2. Decode the base64 audio string and write it to the initial file.
        try:
            with open(temp_input_audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            print(f"Decoded base64 audio to {temp_input_audio_path}")
        except Exception as e:
            return {"error": f"Failed to decode base64 audio: {str(e)}"}

        # 3. Use FFmpeg to "launder" the audio file.
        #    This command tells FFmpeg to auto-detect the input format and convert it
        #    to a standard 16kHz mono WAV file, which is ideal for Whisper.
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', temp_input_audio_path, # Input file
            '-ar', '16000',              # Output sample rate: 16000 Hz
            '-ac', '1',                  # Output channels: 1 (mono)
            '-c:a', 'pcm_s16le',         # Output codec: standard 16-bit PCM
            clean_wav_path               # The clean output WAV file
        ]

        print(f"Running FFmpeg to normalize audio: {' '.join(ffmpeg_cmd)}")
        try:
            # We use a timeout to prevent hung processes
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=30)
            print("FFmpeg audio normalization successful.")
        except subprocess.CalledProcessError as e:
            print("FFmpeg audio normalization failed.")
            print("FFmpeg STDERR:", e.stderr)
            return {"error": "FFmpeg audio normalization failed.", "details": e.stderr}
        except subprocess.TimeoutExpired:
            return {"error": "FFmpeg audio normalization timed out."}

        # --- END OF CORRECTED BLOCK ---

        # 4. Transcribe the CLEAN audio file.
        try:
            audio = whisper.load_audio(clean_wav_path)
            result = whisper.transcribe(whisper_model, audio, language="en") # Change to "hi" for Hindi

            # Flatten the word list for easier processing
            all_words = []
            for segment in result["segments"]:
                for word in segment["words"]:
                    all_words.append({
                        "text": word["text"].strip(),
                        "start": word["start"],
                        "end": word["end"]
                    })

            print(f"Transcription successful. Found {len(all_words)} words.")
            with open(clean_wav_path, "rb") as f:
                clean_audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            return {
                "full_text": result["text"],
                "words": all_words,
                "audio_base64": clean_audio_base64
            }

        except Exception as e:
            # Include the error message in the output for easier debugging
            error_message = str(e)
            print(f"Error during transcription: {error_message}")
            if "ffmpeg" in error_message:
                return {"error": f"Transcription failed: Failed to load audio: {error_message}"}
            return {"error": f"Transcription failed: {error_message}"}

# Register our functions with the RunPod serverless SDK.
runpod.serverless.start({
    "handler": handler,
    "cold_start_callback": cold_start
})