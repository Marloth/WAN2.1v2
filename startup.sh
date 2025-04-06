#!/bin/bash
set -e

echo "Starting WAN2.1 I2V service..."

# Check and download model if needed
echo "Checking for model in persistent storage..."
python3 -c "
import os
from utils import download_model_if_needed

MODEL_ID = 'Wan-AI/Wan2.1-I2V-14B-480P'
MODEL_CACHE_DIR = '/spinvol/models/wan21'

print(f'Ensuring model {MODEL_ID} is downloaded...')
download_model_if_needed(MODEL_ID, MODEL_CACHE_DIR)
print('Model check completed successfully.')
"

# We don't need to start the runpod serverless handler here anymore
# It will be started by the CMD in the Dockerfile

