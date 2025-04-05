FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir runpod

# Copy application files
COPY handler.py .
COPY utils.py .

# Create directories for persistent storage
RUN mkdir -p /runpod-volume/models/wan21
RUN mkdir -p /runpod-volume/cache

# Set environment variables for cache locations
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE="/runpod-volume/cache"
ENV HF_HOME="/runpod-volume/cache"

# Default command
CMD ["python3", "-m", "runpod.serverless.start"]
