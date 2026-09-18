from two_init_and_clone_respawn_final import generate_metadrive_respawn_scene
from two_init_and_clone_hybrid_final import generate_metadrive_hybrid_scene
from three_check import run_check
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
ROOT_PATH = 'your_root_directory'
BASE_PATH = "your_base_directory/output/img_scene"

def parse_triplet(file_path):
    """Parse a triplet file containing map configuration"""
    with open(file_path, 'r') as file:
        content = file.read().strip().strip('()')
        elements = [elem.strip() for elem in content.split(',')]
        if len(elements) != 3:
            raise ValueError("Invalid triplet format")
        return elements[0].strip('\'"'), int(elements[1]), float(elements[2])

def merge_txt_files(output_dir, image_name):
    # Define input file paths
    init_respawn_file = os.path.join(output_dir, image_name, "preparation", f"{image_name}_respawn.txt")
    init_hybrid_file = os.path.join(output_dir, image_name, "preparation", f"{image_name}_hybrid.txt")
    
    # Define output file path
    init_merge_file = os.path.join(output_dir, image_name, "preparation", f"{image_name}_merge.txt")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(init_merge_file), exist_ok=True)
    
    try:
        # Read first file content
        with open(init_respawn_file, 'r') as file1:
            content1 = file1.read()
        
        # Read second file content
        with open(init_hybrid_file, 'r') as file2:
            content2 = file2.read()
        
        # Merge contents
        merged_content = content1 + content2
        
        # Write merged content to new file
        with open(init_merge_file, 'w') as outfile:
            outfile.write(merged_content)
            
        print(f"Files successfully merged to: {init_merge_file}")
        return True
    
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return False
    except Exception as e:
        print(f"Error merging files: {e}")
        return False

# Read road info
img_folder = os.path.join(ROOT_PATH, 'output', 'img_bev')
road_geo_folder = os.path.join(ROOT_PATH, 'output', 'road_geo')
img_name_list = []
output_dir = os.path.join(ROOT_PATH, 'output', 'img_scene')

# Iterate through all files in the folder
for filename in os.listdir(img_folder):
    # Check if file is an image (png/jpg/jpeg)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Extract image name (without extension)
        img_name = os.path.splitext(filename)[0]
        img_name_list.append(img_name)

for image_name in img_name_list:
    print(image_name)
    road_geo_file_path = os.path.join(road_geo_folder, f"{image_name}_road_geo.txt")
    
    map_char, lane_num, max_speed = parse_triplet(road_geo_file_path)

    # Generate hybrid mode scene
    generate_metadrive_hybrid_scene(map_char, lane_num, output_dir, image_name)
    
    # Generate respawn mode scene
    generate_metadrive_respawn_scene(map_char, lane_num, output_dir, image_name)

    # Merge files
    merge_txt_files(output_dir, image_name)
    
    merge_filename = os.path.join(output_dir, image_name, "preparation", f"{image_name}_merge.txt")
    print(merge_filename)
    
    # Run validation check
    run_check(
        input_filename=merge_filename,
        map_char=map_char,
        lane_num=lane_num,
        image_name=image_name,
        root_path=output_dir
    )