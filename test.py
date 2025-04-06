import runpod
import base64
from PIL import Image
import io
import time
import json

# Initialize RunPod client with your API key
runpod.api_key = "rpa_J7ZZZDXEWWPCYX1MRFVURDPIU6J8HWHMWE4YATN87n0ulx"

# Open an image and convert to base64
with open("image.webp", "rb") as image_file:
    image_data = base64.b64encode(image_file.read()).decode('utf-8')

# Create the input payload
payload = {
    "input": {
        "image": image_data,
        "prompt": "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard.",
        "negative_prompt": "poor quality, distortion, blurry",
        "num_inference_steps": 25,
        "size": "512*512"
    }
}

# Initialize the endpoint
endpoint = runpod.Endpoint("5y1ko00m9048uu")

# Send the request to your endpoint
response = endpoint.run(payload)

# Save the generated frames
print("Response received:", response)

# The response is a Job object in the latest RunPod API
print("Waiting for job to complete...")

# Wait for the job to complete and get the output
try:
    # Get the job output with a timeout of 600 seconds (10 minutes)
    result = response.output(timeout=600)
    print("Job completed!")
    print("Result:", result)
    
    # Check if we have frames in the result
    if isinstance(result, dict) and "frames" in result:
        for i, frame_base64 in enumerate(result["frames"]):
            img_data = base64.b64decode(frame_base64)
            with open(f"frame_{i:02d}.png", "wb") as f:
                f.write(img_data)
        print(f"Successfully saved {len(result['frames'])} frames")
    else:
        print("No frames found in the response. Result structure:")
        print(json.dumps(result, indent=2))
        
except Exception as e:
    print(f"Error waiting for job: {e}")