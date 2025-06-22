# whisper_handler.py (Corrected to handle raw PCM audio from Gemini TTS)
import runpod
import os
import base64
import tempfile
import whisper_timestamped as whisper
import json
import subprocess # Make sure subprocess is imported

# Global variable to hold the Whisper model, so we only load it once.
whisper_model = None

def cold_start():
    """
    This function is called once when the worker starts up.
    It's the perfect place to load the heavy Whisper model into memory (and VRAM).
    """
    global whisper_model
    print("Cold start: Loading Whisper model...")
    # Load the same model specified in the Dockerfile. 'medium' is a good choice.
    # We specify device="cuda" to ensure it runs on the GPU.
    whisper_model = whisper.load_model("medium", device="cuda")
    print("Whisper model loaded successfully.")

async def handler(job):
    """
    This is the main handler function that processes each incoming request.
    The 'job' object contains the input from the API call.
    """
    print(f"Received transcription job: {job['id']}")
    job_input = job['input']
    audio_base64 = job_input.get("audio_base64")

    if not audio_base64:
        return {"error": "No 'audio_base64' provided in job input."}

    # Safety check: ensure the model is loaded.
    global whisper_model
    if whisper_model is None:
        cold_start()

    # Use a temporary directory to safely handle file operations.
    with tempfile.TemporaryDirectory(dir="/tmp/whisper_audio") as temp_dir:
        # --- START OF MODIFIED BLOCK ---

        # 1. Define paths for the raw PCM input and the final WAV output
        raw_audio_path = os.path.join(temp_dir, "input_audio.pcm")
        wav_output_path = os.path.join(temp_dir, "input_audio.wav")

        # 2. Decode the base64 audio string and write it to the raw PCM file.
        try:
            with open(raw_audio_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            print(f"Decoded base64 PCM audio to {raw_audio_path}")
        except Exception as e:
            return {"error": f"Failed to decode base64 audio: {str(e)}"}

        # 3. Use FFmpeg to convert the raw PCM data into a proper WAV file.
        #    We must specify the input format details provided by the Gemini API.
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',                   # Overwrite output file if it exists
            '-f', 's16le',          # Input format: signed 16-bit little-endian PCM
            '-ar', '24000',         # Input sample rate: 24000 Hz
            '-ac', '1',             # Input channels: 1 (mono)
            '-i', raw_audio_path,   # The input raw audio file
            wav_output_path         # The output WAV file
        ]

        print(f"Running FFmpeg to convert PCM to WAV: {' '.join(ffmpeg_cmd)}")
        try:
            result = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print("FFmpeg conversion successful.")
        except subprocess.CalledProcessError as e:
            print("FFmpeg PCM-to-WAV conversion failed.")
            print("FFmpeg STDERR:", e.stderr)
            return {"error": "FFmpeg PCM-to-WAV conversion failed.", "details": e.stderr}

        # --- END OF MODIFIED BLOCK ---

        # Perform the transcription using the correctly formatted WAV file.
        try:
            # Load the audio file using whisper's helper function.
            # CRUCIALLY, we load the new WAV file, not the original input.
            audio = whisper.load_audio(wav_output_path)

            # Transcribe the audio. Specify the language if you know it for better accuracy.
            result = whisper.transcribe(whisper_model, audio, language="en") # Change to "hi" for Hindi

            # Flatten the word list for easier processing by the next service.
            all_words = []
            for segment in result["segments"]:
                for word in segment["words"]:
                    all_words.append({
                        "text": word["text"].strip(),
                        "start": word["start"],
                        "end": word["end"]
                    })

            print(f"Transcription successful. Found {len(all_words)} words.")
            
            # Return the final, structured JSON object.
            return {
                "full_text": result["text"],
                "words": all_words
            }

        except Exception as e:
            print(f"Error during transcription: {str(e)}")
            return {"error": f"Transcription failed: {str(e)}"}

# Register our functions with the RunPod serverless SDK.
runpod.serverless.start({
    "handler": handler,
    "cold_start_callback": cold_start
})