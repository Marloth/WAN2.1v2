# Use the NVIDIA PyTorch container as base image - already has PyTorch installed
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Install minimal system dependencies required for Wan2.1 and OpenCV
RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /wan21

# Wan2.1 repository already cloned above

# Clone the Wan2.1 repository first to get their requirements
RUN git clone https://github.com/Wan-Video/Wan2.1.git /wan21/wan2_repo

# Install Python dependencies in a single step to reduce layers
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir runpod "huggingface_hub[cli]" && \
    pip install --no-cache-dir transformers diffusers accelerate safetensors && \
    pip install --no-cache-dir pillow numpy opencv-python ffmpeg-python

# Install the requirements from the Wan2.1 repo
RUN cd /wan21/wan2_repo
RUN pip install --no-cache-dir -e .

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
