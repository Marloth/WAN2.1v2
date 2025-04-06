import os
import time
import torch
import base64
import runpod
import tempfile
import logging
import subprocess
import glob
from io import BytesIO
from PIL import Image
from utils import download_model_if_needed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directories to use persistent storage
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/cache"
os.environ["HF_HOME"] = "/runpod-volume/cache"

# Global variables
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
# Use the new storage volume with more space
MODEL_CACHE_DIR = "/spinvol/models/wan21"

# No need to keep the model in memory since we're using the official script

def check_model():
    """Check if the WAN2.1 model is available in persistent storage."""
    logger.info("Checking for WAN2.1 model...")
    
    # Make sure model is in persistent storage
    download_model_if_needed(MODEL_ID, MODEL_CACHE_DIR)
    
    logger.info(f"Model check completed successfully")
    return True

def handler(event):
    """
    RunPod handler function that processes image-to-video generation requests using official Wan2.1 implementation.
    """
    # Make sure model is in persistent storage
    check_model()
    
    try:
        start_time = time.time()
        logger.info("Processing request...")
        
        # Get input parameters with defaults
        job_input = event["input"]
        
        # Validate required inputs
        if "image" not in job_input:
            return {"error": "Input image is required"}
        
        # Extract parameters
        prompt = job_input.get("prompt", "")
        negative_prompt = job_input.get("negative_prompt", "poor quality, distortion")
        num_inference_steps = job_input.get("num_inference_steps", 25)
        size = job_input.get("size", "512*512")
        
        # Process input image
        logger.info("Decoding input image...")
        image_base64 = job_input["image"]
        image_data = base64.b64decode(image_base64)
        
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the input image
            input_image_path = os.path.join(temp_dir, "input_image.jpg")
            with open(input_image_path, "wb") as f:
                f.write(image_data)
            
            # Output directory for frames
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Set up the command to run the official generate.py script
            generate_script = "/wan21/wan2_repo/generate.py"
            
            # Build the command
            cmd = [
                "python", generate_script,
                "--task", "i2v-14B", 
                "--size", size,
                "--ckpt_dir", MODEL_CACHE_DIR,
                "--image", input_image_path,
                "--prompt", prompt,
                "--num_inference_steps", str(num_inference_steps),
                "--num_frames", "17",  # Always use 17 frames
                "--output_dir", output_dir
            ]
            
            if negative_prompt:
                cmd.extend(["--negative_prompt", negative_prompt])
            
            # Run the official Wan2.1 generate script
            logger.info(f"Running generate command: {' '.join(cmd)}")
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode != 0:
                error_msg = process.stderr or "Unknown error running generation"
                logger.error(f"Error running generation: {error_msg}")
                return {"error": error_msg}
            
            # Find the generated frame images (they should be in the output directory)
            frame_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))
            
            if not frame_paths:
                return {"error": "No frames were generated"}
            
            logger.info(f"Found {len(frame_paths)} generated frames")
            
            # Ensure we have exactly 17 frames by padding or truncating
            if len(frame_paths) < 17:
                logger.warning(f"Only got {len(frame_paths)} frames, will pad to 17 frames")
                # We'll handle this in the frame encoding loop
            elif len(frame_paths) > 17:
                logger.warning(f"Got {len(frame_paths)} frames, truncating to 17 frames")
                frame_paths = frame_paths[:17]
            
            # Encode each frame as base64 PNG
            frame_images = []
            
            # First encode all available frames
            for frame_path in frame_paths:
                with open(frame_path, "rb") as f:
                    frame_data = f.read()
                    img_str = base64.b64encode(frame_data).decode("utf-8")
                    frame_images.append(img_str)
            
            # If we have fewer than 17 frames, duplicate the last frame
            if len(frame_images) < 17:
                last_frame = frame_images[-1]
                while len(frame_images) < 17:
                    frame_images.append(last_frame)
            
            logger.info(f"Encoded {len(frame_images)} PNG frames to base64")
        
        # Return the results
        total_time = time.time() - start_time
        logger.info(f"Request processed successfully in {total_time:.2f} seconds")
        
        return {
            "frames": frame_images,
            "frame_count": len(frame_images),
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": 17,  # Always 17 frames
                "size": size,
                "num_inference_steps": num_inference_steps
            },
            "metrics": {
                "processing_time": f"{total_time:.2f}s"
            }
        }
        
    except Exception as e:
        # Log error details
        import traceback
        error_message = str(e)
        error_traceback = traceback.format_exc()
        logger.error(f"Error processing request: {error_message}\n{error_traceback}")
        
        # Return error information
        return {
            "error": error_message,
            "traceback": error_traceback
        }

# Start the serverless function
runpod.serverless.start({"handler": handler})
