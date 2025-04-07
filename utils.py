import os
import tempfile
import subprocess
import logging
import sys
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dependencies():
    """
    Check for and install required dependencies if they're missing.
    """
    required_packages = [
        'easydict',
        'einops',
        'torch-fidelity',
        'imageio',
        'imageio-ffmpeg',
        'av',
        'clip',
        'dashscope',
        'flash-attn>=2.0.0,<3.0.0'  # Flash Attention 2.x is required by Wan2.1
    ]
    
    for package in required_packages:
        try:
            # Try to import the package
            importlib.import_module(package.replace('-', '_'))
            logger.info(f"✅ {package} is already installed")
        except ImportError:
            # If import fails, install the package
            logger.info(f"⚠️ {package} is missing, installing...")
            
            # Special case for CLIP which requires git installation
            if package == 'clip':
                pip_cmd = "git+https://github.com/openai/CLIP.git"
            else:
                pip_cmd = package
                
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_cmd])
                logger.info(f"✅ Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Failed to install {package}: {str(e)}")
                # Continue trying other packages rather than failing completely

def download_model_if_needed(model_id, cache_dir):
    """
    Checks if model exists in persistent storage and downloads only if needed.
    
    Args:
        model_id: Hugging Face model ID
        cache_dir: Directory to store the model
    
    Returns:
        Path to the model directory
    """
    # Check if model already exists in persistent storage
    model_index_path = os.path.join(cache_dir, "model_index.json")
    
    if not os.path.exists(model_index_path):
        logger.info(f"Model not found in persistent storage. Downloading {model_id} to {cache_dir}...")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download model from Hugging Face using huggingface-cli
        try:
            cmd = [
                "huggingface-cli", "download",
                model_id,
                "--local-dir", cache_dir
            ]
            
            # Run the huggingface-cli command
            process = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Model downloaded successfully to {cache_dir}")
            logger.debug(process.stdout)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            logger.error(f"Error downloading model: {error_msg}")
            raise
    else:
        logger.info(f"Model already exists in persistent storage at {cache_dir}")
    
    return cache_dir

def create_video_from_frames(frames, output_path, fps=8):
    """
    Creates a video from a list of PIL Image frames using ffmpeg.
    
    Args:
        frames: List of PIL Image objects
        output_path: Path to save the output video
        fps: Frames per second for the output video
    
    Returns:
        Path to the created video file
    """
    logger.info(f"Creating video with {len(frames)} frames at {fps} fps")
    
    # Create a temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save frames as images
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            frame.save(frame_path)
            frame_paths.append(frame_path)
        
        logger.info(f"Saved {len(frame_paths)} frames to temporary directory")
        
        # Use ffmpeg to create a video
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", os.path.join(temp_dir, "frame_%04d.png"),
            "-c:v", "libx264",
            "-profile:v", "high",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            output_path
        ]
        
        # Run the ffmpeg command
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Video created successfully at {output_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating video: {e.stderr.decode() if e.stderr else str(e)}")
            raise
    
    return output_path
