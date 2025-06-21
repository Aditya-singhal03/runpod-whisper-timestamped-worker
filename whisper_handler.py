# whisper_handler.py
import runpod
import os
import base64
import tempfile
import whisper_timestamped as whisper
import json

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
        # Create a path for the temporary audio file.
        audio_input_path = os.path.join(temp_dir, "input_audio.wav")

        # Decode the base64 audio string and write it to the file.
        try:
            with open(audio_input_path, "wb") as f:
                f.write(base64.b64decode(audio_base64))
            print(f"Decoded base64 audio to {audio_input_path}")
        except Exception as e:
            return {"error": f"Failed to decode base64 audio: {str(e)}"}

        # Perform the transcription to get word-level timestamps.
        try:
            # Load the audio file using whisper's helper function.
            audio = whisper.load_audio(audio_input_path)

            # Transcribe the audio. Specify the language if you know it for better accuracy.
            result = whisper.transcribe(whisper_model, audio, language="en") # Change to "hi" for Hindi

            # The 'result' object is already in the JSON format we want.
            # We can optionally "flatten" the word list for easier processing by the next service.
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