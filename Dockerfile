# Dockerfile
# Use a RunPod PyTorch base image with CUDA for GPU acceleration.
# Whisper is significantly faster on a GPU.
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Set environment variables for non-interactive installation
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install FFmpeg, which is a required dependency for Whisper to process audio.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set the working directory inside the container
WORKDIR /rp_handler

# Copy your Python requirements file and your handler script
COPY requirements.txt .
COPY whisper_handler.py .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- Pre-download the Whisper model ---
# This is the most important optimization for a fast cold start.
# The 'medium' model is a great balance of accuracy and speed. You can also use 'base', 'small', or 'large'.
# This command downloads the model into the Docker image's cache.
RUN python -c "import whisper; whisper.load_model('medium')"

# Create a temporary directory for audio files that will be processed
RUN mkdir -p /tmp/whisper_audio