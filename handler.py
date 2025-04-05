import os
import time
import torch
import base64
import runpod
import tempfile
import logging
from io import BytesIO
from PIL import Image
from utils import download_model_if_needed, create_video_from_frames

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set cache directories to use persistent storage
os.environ["TRANSFORMERS_CACHE"] = "/runpod-volume/cache"
os.environ["HF_HOME"] = "/runpod-volume/cache"

# Global variables
MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
MODEL_CACHE_DIR = "/runpod-volume/models/wan21"

# Model will be loaded on first request
model = None

def load_model():
    """Load the WAN2.1 model from persistent storage."""
    logger.info("Loading WAN2.1 model...")
    start_time = time.time()
    
    # Import here to reduce cold start time
    from diffusers import DiffusionPipeline
    
    # Make sure model is in persistent storage
    download_model_if_needed(MODEL_ID, MODEL_CACHE_DIR)
    
    # Load model in FP16 for better memory efficiency
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_CACHE_DIR, 
        torch_dtype=torch.float16,
        local_files_only=True
    )
    pipe.to("cuda")
    
    logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    return pipe

def handler(event):
    """
    RunPod handler function that processes image-to-video generation requests.
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        model = load_model()
    
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
        num_frames = job_input.get("num_frames", 16)
        fps = job_input.get("fps", 8)
        num_inference_steps = job_input.get("num_inference_steps", 25)
        
        # Process input image
        logger.info("Decoding input image...")
        image_base64 = job_input["image"]
        image_data = base64.b64decode(image_base64)
        init_image = Image.open(BytesIO(image_data)).convert("RGB")
        
        # Ensure image has appropriate dimensions
        orig_width, orig_height = init_image.size
        if init_image.width % 8 != 0 or init_image.height % 8 != 0:
            # Resize to nearest multiple of 8
            width = (init_image.width // 8) * 8
            height = (init_image.height // 8) * 8
            init_image = init_image.resize((width, height))
            logger.info(f"Resized image from {orig_width}x{orig_height} to {width}x{height}")
        
        # Generate video frames
        logger.info(f"Generating video with {num_frames} frames...")
        with torch.autocast("cuda"):
            result = model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
            )
        
        logger.info(f"Video frames generated in {time.time() - start_time:.2f} seconds")
        
        # Create temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Get the generated frames
            frames = result.frames[0]  # WAN2.1 returns frames in this format
            
            logger.info(f"Creating video file with {len(frames)} frames at {fps} fps...")
            
            # Create a video file from the frames
            output_path = os.path.join(temp_dir, f"output_{int(time.time())}.mp4")
            create_video_from_frames(frames, output_path, fps)
            
            # Read and encode the video file
            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()
                video_base64 = base64.b64encode(video_bytes).decode("utf-8")
                logger.info(f"Video encoded to base64, size: {len(video_base64) // 1024} KB")
        
        # Return the results
        total_time = time.time() - start_time
        logger.info(f"Request processed successfully in {total_time:.2f} seconds")
        
        return {
            "video_base64": video_base64,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "fps": fps,
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
