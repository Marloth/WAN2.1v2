FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip git wget ffmpeg libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /wan21

# Clone the Wan2.1 repository
RUN git clone https://github.com/Wan-Video/Wan2.1.git /wan21/wan2_repo

# Install Python dependencies
COPY requirements.txt .
# Ensure torch >= 2.4.0
RUN pip3 install --no-cache-dir -r requirements.txt
# Install dependencies from the cloned repo
RUN pip3 install --no-cache-dir -r /wan21/wan2_repo/requirements.txt
RUN pip3 install --no-cache-dir runpod
RUN pip3 install --no-cache-dir "huggingface_hub[cli]"

# Copy application files
COPY handler.py .
COPY utils.py .
COPY startup.sh .

# Make startup script executable
RUN chmod +x startup.sh

# Create directories for persistent storage
RUN mkdir -p /runpod-volume/models/wan21
RUN mkdir -p /runpod-volume/cache

# Set environment variables for cache locations
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE="/runpod-volume/cache"
ENV HF_HOME="/runpod-volume/cache"

# Use startup script as entrypoint (will download model if needed)
CMD ["/wan21/startup.sh"]
