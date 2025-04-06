import os
import sys
import base64
import shutil
import subprocess
import json
import tempfile
import time
from PIL import Image
import io
import uuid
import runpod
from utils import download_model_if_needed, ensure_dependencies
import glob
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables
os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:/wan21/wan2_repo"

# If CUDA_VISIBLE_DEVICES isn't set, set it to 0
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Global variables
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
# Use the default network volume path
MODEL_CACHE_DIR = "/runpod-volume/models/wan21"

def handler(job):
    """
    RunPod handler function that processes image-to-video generation requests using official Wan2.1 implementation.
    """
    try:
        start_time = time.time()
        logger.info("Processing request...")
        
        # Get input parameters with defaults
        job_input = job["input"]
        
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
            
            # Build the command exactly matching the example from documentation
            cmd = [
                "python", generate_script,
                "--task", "i2v-14B",
                "--size", size,
                "--ckpt_dir", MODEL_CACHE_DIR,
                "--image", input_image_path,
                "--prompt", prompt,
                "--use_prompt_extend", "False",
                "--save_frames",
                "--out_dir", output_dir
            ]
            
            # Add steps parameter if provided (optional)
            if num_inference_steps:
                cmd.extend(["--steps", str(num_inference_steps)])
            
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
            logger.info(f"Looking for generated frames in: {output_dir}")
            frame_paths = sorted(glob.glob(os.path.join(output_dir, "*.png")))
            
            # Also check parent directory and subdirectories in case frames are saved elsewhere
            if not frame_paths:
                logger.info("No frames found in output directory, checking parent directory")
                frame_paths = sorted(glob.glob(os.path.join(os.path.dirname(output_dir), "*.png")))
            
            if not frame_paths:
                logger.info("No frames found in parent directory, checking subdirectories")
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        if file.endswith(".png"):
                            frame_paths.append(os.path.join(root, file))
                frame_paths = sorted(frame_paths)
            
            if not frame_paths:
                logger.info(f"Directory contents of {output_dir}: {os.listdir(output_dir) if os.path.exists(output_dir) else 'directory does not exist'}")
                logger.info(f"Directory contents of parent: {os.listdir(os.path.dirname(output_dir)) if os.path.exists(os.path.dirname(output_dir)) else 'directory does not exist'}")
                return {"error": "No frames were generated"}
            
            logger.info(f"Found {len(frame_paths)} generated frames at: {frame_paths[0] if frame_paths else 'None'}")
            
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

# This is the proper way to start a RunPod serverless handler
if __name__ == "__main__":
    # Check and install any missing dependencies
    logger.info("Checking for required dependencies...")
    ensure_dependencies()
    logger.info("Dependency check completed")
    
    # Check if model is available before starting server
    logger.info("Ensuring model is downloaded before starting serverless handler...")
    download_model_if_needed(MODEL_ID, MODEL_CACHE_DIR)
    logger.info("Model check completed, starting serverless handler")
    
    # Start the serverless worker with our handler
    runpod.serverless.start({"handler": handler})
