# whisper_handler.py (Corrected to properly handle raw PCM input from Gemini)
import runpod
import os
import base64
import tempfile
import whisper_timestamped as whisper
import json
import subprocess

# Global variable to hold the Whisper model
whisper_model = None

def cold_start():
    """Loads the Whisper model into memory (and VRAM)."""
    global whisper_model
    print("Cold start: Loading Whisper model...")
    whisper_model = whisper.load_model("medium", device="cuda")
    print("Whisper model loaded successfully.")

async def handler(job):
    """Processes each incoming transcription request."""
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

        # 1. Define paths for the raw PCM input and the final WAV output
        raw_audio_path = os.path.join(temp_dir, "input.pcm")
        clean_wav_path = os.path.join(temp_dir, "output.wav")

        # 2. Decode the base64 audio and save it to the raw PCM file.
        try:
            with open(raw_audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            print(f"Decoded base64 PCM audio to {raw_audio_path}")
        except Exception as e:
            return {"error": f"Failed to decode base64 audio: {str(e)}"}

        # 3. Use FFmpeg to convert the raw PCM data into a proper WAV file.
        #    This command explicitly describes the raw input format.
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',                   # Overwrite output file
            '-f', 's16le',          # INPUT format: signed 16-bit little-endian PCM
            '-ar', '24000',         # INPUT sample rate: 24000 Hz
            '-ac', '1',             # INPUT channels: 1 (mono)
            '-i', raw_audio_path,   # The input raw audio file
            clean_wav_path          # The output WAV file
        ]

        print(f"Running FFmpeg to convert PCM to WAV: {' '.join(ffmpeg_cmd)}")
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=30)
            print("FFmpeg PCM-to-WAV conversion successful.")
        except subprocess.CalledProcessError as e:
            print("FFmpeg PCM-to-WAV conversion failed.")
            print("FFmpeg STDERR:", e.stderr)
            return {"error": "FFmpeg PCM-to-WAV conversion failed.", "details": e.stderr}
        except subprocess.TimeoutExpired:
            return {"error": "FFmpeg PCM-to-WAV conversion timed out."}

        # --- END OF CORRECTED BLOCK ---

        # 4. Transcribe the CLEAN audio file.
        try:
            # Check if the output WAV file was created and is not empty
            if not os.path.exists(clean_wav_path) or os.path.getsize(clean_wav_path) == 0:
                raise RuntimeError("FFmpeg created an empty or missing WAV file.")

            audio = whisper.load_audio(clean_wav_path)
            result = whisper.transcribe(model=whisper_model, audio=audio, language="en") # Change to "hi" for Hindi

            # Flatten the word list for easier processing
            all_words = []
            for segment in result["segments"]:
                for word in segment["words"]:
                    all_words.append({
                        "text": word["text"].strip(),
                        "start": word["start"],
                        "end": word["end"]
                    })
            
            # Read the clean WAV file and encode it to base64 to return it
            with open(clean_wav_path, "rb") as f:
                clean_audio_base64 = base64.b64encode(f.read()).decode('utf-8')

            print(f"Transcription successful. Found {len(all_words)} words.")
            
            return {
                "full_text": result["text"],
                "words": all_words,
                "audio_base64": clean_audio_base64
            }

        except Exception as e:
            error_message = str(e)
            print(f"Error during transcription: {error_message}")
            return {"error": f"Transcription failed: {error_message}"}

# Register the handler with RunPod
runpod.serverless.start({
    "handler": handler,
    "cold_start_callback": cold_start
})