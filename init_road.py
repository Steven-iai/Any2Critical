import os
import math
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.map.base_map import BaseMap
from metadrive.component.map.pg_map import MapGenerateMethod
from metadrive.constants import DEFAULT_AGENT
from metadrive.utils.doc_utils import generate_gif
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.vehicle.vehicle_type import DefaultVehicle
import sys
from PIL import Image
import numpy as np
import cv2
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
ROOT_PATH = 'your_project_root_directory'

def parse_triplet(file_path):
    """Parse configuration file containing map parameters as a triplet"""
    with open(file_path, 'r') as file:
        content = file.read().strip().strip('()')
        elements = [elem.strip() for elem in content.split(',')]
        if len(elements) != 3:
            raise ValueError("Invalid triplet format")
        return elements[0].strip('\'"'), int(elements[1]), float(elements[2])

def generate_metadrive_lane_scene(map_char, lane_num, output_dir, image_name):
    """
    Generate MetaDrive scene and save vehicle information (respawn mode)
    
    Args:
        map_char: Map character (e.g. "C")
        lane_num: Number of lanes
        output_dir: Output directory path
        image_name: Scene name (used for filenames)
    """
    env = MetaDriveEnv(
        dict(
            map_config={
                BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_SEQUENCE,
                BaseMap.GENERATE_CONFIG: map_char,
                BaseMap.LANE_WIDTH: 3.5,
                BaseMap.LANE_NUM: lane_num
            },
            log_level=50,
            traffic_density=0,
            traffic_mode="respawn",
            random_spawn_lane_index=False,
            agent_configs={
                DEFAULT_AGENT: dict(
                    use_special_color=True,
                    spawn_position_heading=([0, 0], 0)
                )
            }
        )
    )

    try:
        env.reset()
        frames = []
        for step in range(2):
            obs, _, done, _, _ = env.step([0, 0])
            frame = env.render(
                mode="topdown", 
                window=False,
                screen_size=(2000, 2000),
                camera_position=(60, 15)
            )
            frames.append(frame)
            if step == 1:
                generate_gif(
                    frames, 
                    gif_name=os.path.join(output_dir, image_name, "preparation", f"{image_name}_meta_lane.gif")
                )
                print(f"Data saved to: {os.path.join(output_dir, image_name)}")
    finally:
        env.close()

def save_masked_image(
    image_pil: Image.Image,
    masks: np.ndarray,
    output_dir: str,
    output_filename: str,
    refine: bool = True
):
    """Save only the road-masked area of the original image, cropping out the rest"""
    os.makedirs(output_dir, exist_ok=True)
    image = image_pil.convert("RGBA")
    image_np = np.array(image)
    
    combined_mask = np.zeros(masks[0].squeeze().shape, dtype=np.uint8)
    for mask in masks:
        combined_mask = np.logical_or(combined_mask, mask.squeeze())
    
    if refine:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(combined_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        filled = closed.copy()
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(filled, contours, -1, 1, -1)
        smoothed = cv2.medianBlur(filled, 7)
        combined_mask = cv2.dilate(smoothed, np.ones((5, 5), np.uint8), iterations=1)
    
    result_np = np.zeros_like(image_np)
    result_np[combined_mask > 0] = image_np[combined_mask > 0]
    
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    cropped_result = result_np[ymin:ymax+1, xmin:xmax+1]
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
    text_prompt_road = "road."
    
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

img_folder = os.path.join(ROOT_PATH, 'output', 'img_bev')
road_geo_folder = os.path.join(ROOT_PATH, 'output', 'road_geo')
img_name_list = []
output_dir = os.path.join(ROOT_PATH, 'output', 'img_scene')

metadrive_screenshot_dir = os.path.join(ROOT_PATH, "output", "metadrive_screenshot")
metadrive_lane_dir = os.path.join(ROOT_PATH, "output", "metadrive_lane")
os.makedirs(metadrive_screenshot_dir, exist_ok=True)
os.makedirs(metadrive_lane_dir, exist_ok=True)

for filename in os.listdir(img_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_name = os.path.splitext(filename)[0]
        img_name_list.append(img_name)

for image_name in img_name_list:
    road_geo_file_path = os.path.join(road_geo_folder, f"{image_name}_road_geo.txt")
    map_char, lane_num, max_speed = parse_triplet(road_geo_file_path)
    generate_metadrive_lane_scene(map_char, lane_num, output_dir, image_name)
    gif_path = os.path.join(output_dir, image_name, "preparation", f"{image_name}_meta_lane.gif")
    
    try:
        with open(gif_path, "rb") as f:
            print("File can be opened manually!")
    except Exception as e:
        print(f"Failed to open manually: {e}")
    
    try:
        with open(gif_path, "rb") as f:
            header = f.read(6)
            print(f"File header: {header}")
    except Exception as e:
        print(f"Failed to read file header: {e}")

    output_path = os.path.join(ROOT_PATH, "output", "metadrive_screenshot", f"{image_name}.png")
  
    try:
        with Image.open(gif_path) as gif:
            first_frame = gif.convert('RGBA')
            first_frame.save(output_path, 'PNG')
            print(f"Successfully saved first frame to: {output_path}")
    except FileNotFoundError:
        print(f"Error: GIF file not found {gif_path}")
        print("Retrying...")
        with Image.open(gif_path) as gif:
            first_frame = gif.convert('RGBA')
            first_frame.save(output_path, 'PNG')
            print(f"Successfully saved first frame to: {output_path}")
    except Exception as e:
        print(f"Error during processing: {str(e)}")