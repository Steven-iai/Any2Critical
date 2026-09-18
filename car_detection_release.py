import json
import time
import requests
import base64

def png_to_base64(file_path):
    """
    Convert a PNG image file to a Base64 encoded string
    
    Args:
        file_path (str): Path to the PNG image file
    
    Returns:
        str: Base64 encoded string
    """
    try:
        with open(file_path, "rb") as image_file:
            # Read image binary data
            image_data = image_file.read()
            
            # Encode binary data to Base64 bytes
            base64_bytes = base64.b64encode(image_data)
            
            # Convert bytes to string (if string output is needed)
            base64_string = base64_bytes.decode('utf-8')
            
            return base64_string
            
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return None
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

# Example usage
image_path = 'your_image_file_path_here'  # Replace with your actual image path
base64_str = png_to_base64(image_path)

# 1. Prepare headers with Token
headers = {
    "Content-Type": "application/json",
    "Token": "your-token"
}

# 2. Make algorithm API call
resp = requests.post(
    url='https://api.deepdataspace.com/v2/task/grounding_dino/detection',  # API endpoint
    json={
        "model": "GroundingDino-1.6-Pro",
        "image": "data:image/png;base64,"+base64_str,
        "prompt": {
            "type": "text",
            "text": "car"
        },
        "targets": ["bbox"],
        "bbox_threshold": 0.25,
        "iou_threshold": 0.8
    },
    headers=headers
)
json_resp = resp.json()
print(json_resp)

# 3. Get task_uuid from response
task_uuid = json_resp["data"]["task_uuid"]

# 4. Poll task status
while True:
    resp = requests.get(f'https://api.deepdataspace.com/v2/task_status/{task_uuid}', headers=headers)
    json_resp = resp.json()
    if json_resp["data"]["status"] not in ["waiting", "running"]:
        break
    time.sleep(1)

if json_resp["data"]["status"] == "failed":
    print(json_resp)
elif json_resp["data"]["status"] == "success":
    print(json_resp)