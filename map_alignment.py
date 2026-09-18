import os
import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
ROOT_PATH = 'your_project_root_directory'

def save_masked_image(
    image_pil: Image.Image,
    masks: np.ndarray,
    output_dir: str,
    output_filename: str,
    refine: bool = True
):
    """Save only the road-masked area of original image, cropping out the rest"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to RGBA mode
    image = image_pil.convert("RGBA")
    image_np = np.array(image)
    
    # Combine all masks into one
    combined_mask = np.zeros(masks[0].squeeze().shape, dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask.squeeze())
    
    # Refine the combined mask
    if refine:
        # 1. Fill small holes and gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # 2. Fill all internal holes
        filled = closed.copy()
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(filled, contours, -1, 1, -1)
        
        # 3. Smooth edges
        smoothed = cv2.medianBlur(filled, 7)
        
        # 4. Slightly expand to ensure full coverage
        combined_mask = cv2.dilate(smoothed, np.ones((5, 5), np.uint8), iterations=1)
    
    # Create transparent background
    result_np = np.zeros_like(image_np)
    result_np[combined_mask > 0] = image_np[combined_mask > 0]
    
    # Find mask bounding box
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Crop to bounding box
    cropped_result = result_np[ymin:ymax+1, xmin:xmax+1]
    
    # Save as PNG to preserve transparency
    output_filename = os.path.splitext(output_filename)[0] + '.png'
    output_path = os.path.join(output_dir, output_filename)
    
    Image.fromarray(cropped_result).save(output_path)
    print(f"Saved cropped road mask to: {output_path}")

def process_images(input_dir, output_dir):
    """Process all images in input directory, extracting only road portions"""
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(supported_formats)]
    
    if not image_files:
        print(f"No supported image files found in directory {input_dir}")
        return
    
    model = LangSAM()
    text_prompt_road = "lane."
    
    for img_file in image_files:
        try:
            img_path = os.path.join(input_dir, img_file)
            image_pil = Image.open(img_path).convert("RGB")
            
            road_results = model.predict([image_pil], [text_prompt_road])
            if road_results and len(road_results[0]["masks"]) > 0: 
                save_masked_image(
                    image_pil, 
                    road_results[0]["masks"], 
                    output_dir,
                    img_file
                )
            else:
                print(f"No road detected in image {img_file}")
                
        except Exception as e:
            print(f"Error processing image {img_file}: {str(e)}")

def process_images_and_coordinates(img_a_path, img_b_path, coord_txt_path, output_folder_txt, output_folder_img):
    """Process image pairs and their coordinates"""
    os.makedirs(output_folder_txt, exist_ok=True)
    os.makedirs(output_folder_img, exist_ok=True)
    
    img_a = Image.open(img_a_path)
    img_b = Image.open(img_b_path)
    
    xa, ya = img_a.size
    xb, yb = img_b.size
    
    m = xa / ya
    n = xb / yb
    
    diff_n = abs(n - m)
    diff_reciprocal_n = abs(1/n - m)