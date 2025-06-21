# Dockerfile
# Use a RunPod PyTorch base image with CUDA for GPU acceleration.
# Whisper is significantly faster on a GPU.
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel

# Set environment variables for non-interactive installation and unbuffered Python output
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install FFmpeg and other build tools for a more robust installation process.
# 'build-essential' can help if any pip packages need to be compiled from source.
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set the working directory inside the container
WORKDIR /rp_handler

# Copy your Python requirements file and your handler script
COPY requirements.txt .
COPY whisper_handler.py .

# Install the Python dependencies. We use the --mount cache for faster re-builds.
# This makes subsequent builds much faster if requirements.txt hasn't changed.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# --- Pre-download the Whisper model ---
# This is the most important optimization for a fast cold start.
# The 'medium' model is a great balance of accuracy and speed.
RUN python -c "from whisper import load_model; load_model('medium')"

# Create a temporary directory for audio files that will be processed
RUN mkdir -p /tmp/whisper_audio

# --- Specify the command to run when the container starts ---
# The '-u' flag ensures logs are sent immediately (unbuffered).
CMD ["python", "-u", "whisper_handler.py"]
