from openai import OpenAI
import time
import os
from PIL import Image
import csv
start_whole_time = time.time()
ROOT_PATH = 
SCENARIO_FILEPATH = 
image_folder = 

road_geo_folder = os.path.join(ROOT_PATH, 'output', 'road_geo')
img_scene_folder = os.path.join(ROOT_PATH, 'output', 'img_scene')
img_pos_txt_folder = os.path.join(ROOT_PATH, "output", "car_detection", "aligned")
img_pos_img_folder = os.path.join(ROOT_PATH, "output", "img_aligned")


def parse_triplet(file_path):
    """Parse triplet file containing map configuration"""
    with open(file_path, 'r') as file:
        content = file.read().strip().strip('()')
        elements = [elem.strip() for elem in content.split(',')]
        if len(elements) != 3:
            raise ValueError("Invalid triplet format")
        return elements[0].strip('\'"'), int(elements[1]), float(elements[2])

def file_read(path):
    """
    Read txt file and return:
    - List of lines (automatically removes newlines)
    - Concatenated string (preserves line structure)
    """
    with open(path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]
        data_str = "\n".join(lines)
        return data_str

def extract_valid_five_tuple(file_path):
    """
    Extract multi-tuple data from the last few lines of a file
    
    Args:
        file_path (str): File path
        
    Returns:
        list: List containing multi-tuples, each in format (float, float, float, str, str)
    """
    extracted_tuples = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line in reversed(lines):
            stripped_line = line.strip()
            
            if not stripped_line.startswith('('):
                break
                
            if stripped_line.endswith(')'):
                content = stripped_line[1:-1]
                parts = [part.strip() for part in content.split(',')]
                
                if len(parts) == 5:
                    try:
                        tuple_data = (
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            parts[3],
                            parts[4]
                        )
                        extracted_tuples.append(tuple_data)
                    except ValueError:
                        continue
    
    return extracted_tuples[::-1]

def extract_tuples_from_content(content):
    """
    Extract multi-tuple data from LLM response string
    
    Args:
        content (str): Text content from LLM response
        
    Returns:
        list: List containing multi-tuples, each in format (float, float, float, str, str)
    """
    extracted_tuples = []
    lines = content.split('\n')
    _cnt = 0
    
    for line in reversed(lines):
        stripped_line = line.strip()
        
        if not stripped_line.startswith('('):
            if _cnt>0:
                break
            
        if stripped_line.endswith(')'):
            _cnt = _cnt+1
            content_part = stripped_line[1:-1]
            parts = [part.strip() for part in content_part.split(',')]
            
            if len(parts) == 5:
                try:
                    tuple_data = (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        parts[3],
                        parts[4]
                    )
                    extracted_tuples.append(tuple_data)
                except ValueError:
                    continue
    
    return extracted_tuples[::-1]

def call_llm_for_init_pos(all_available_pos, real_pos, map_char, lane_num, max_speed, width, height, llm_response_init_info_file, init_vehilce_pos_file):
    client = OpenAI(
        base_url='url',
        api_key='key'
    )

    start = time.time()
    
    prompt ="fill in your prompt"

    response = client.chat.completions.create(
        model=model
        messages=[
            {"role": "system", "content": "You are an expert in mapping real-world coordinates to simulator coordinates for autonomous vehicles."},
            {"role": "user", "content": prompt}
        ],
    )

    content = response.choices[0].message.content

    with open(llm_response_init_info_file, 'w', encoding='utf-8') as file:
        file.write(content)

    end = time.time()

    init_list_five_tuple = extract_tuples_from_content(content)

    with open(init_vehilce_pos_file, 'w') as file:
        for tuple_data in init_list_five_tuple:
            file.write(str(tuple_data) + '\n')

    return init_list_five_tuple

import json
import math

def read_vehicle_data(file_path):
    vehicles = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()[1:-1]
            parts = [p.strip().strip("'") for p in line.split(',')]
            if len(parts) == 5:
                x, y, theta = float(parts[0]), float(parts[1]), float(parts[2])
                str1, str2 = parts[3], parts[4]
                vehicles.append({'x': x, 'y': y, 'theta': theta, 'str1': str1, 'str2': str2})
    return vehicles

def rotate_point(x, y, theta_deg):
    """Convert point (x,y) from global coordinates to vehicle-centered local coordinates"""
    theta_rad = math.radians(theta_deg)
    x_rot = math.cos(-theta_rad) * x - math.sin(-theta_rad) * y
    y_rot = math.sin(-theta_rad) * x + math.cos(-theta_rad) * y
    return x_rot, y_rot

def categorize_relative_position(x, y):
    directions = {
        "front":   lambda x, y: x > 0 and abs(y) <= 1,
        "back":    lambda x, y: x < 0 and abs(y) <= 1,
        "left":    lambda x, y: y > 0 and abs(x) <= 1,
        "right":   lambda x, y: y < 0 and abs(x) <= 1,
        "left_front":  lambda x, y: x > 0 and y > 0,
        "right_front": lambda x, y: x > 0 and y < 0,
        "left_back":   lambda x, y: x < 0 and y > 0,
        "right_back":  lambda x, y: x < 0 and y < 0,
    }
    for direction, condition in directions.items():
        if condition(x, y):
            return direction
    return None

def analyze_vehicles(vehicles):
    result = []
    for idx, vehicle in enumerate(vehicles):
        vx, vy, vtheta = vehicle['x'], vehicle['y'], vehicle['theta']
        directions_map = {
            "front": [], "back": [], "left": [], "right": [],
            "left_front": [], "right_front": [], "left_back": [], "right_back": []
        }
        for jdx, other in enumerate(vehicles):
            if idx == jdx:
                continue
            dx = other['x'] - vx
            dy = other['y'] - vy
            rel_x, rel_y = rotate_point(dx, dy, vtheta)
            direction = categorize_relative_position(rel_x, rel_y)
            if direction:
                same_road = 1 if (other['str1'] == vehicle['str1'] and other['str2'] == vehicle['str2']) else 0
                other_with_same_road = other.copy()
                other_with_same_road['same_road'] = same_road
                directions_map[direction].append(other_with_same_road)
        result.append({
            "vehicle": vehicle,
            "directions": directions_map
        })
    return result

import random
import math

def random_choose_vehicles(init_vehicle_pos_list):
    if not init_vehicle_pos_list:
        return []
    
    def calculate_total_distance(point, all_points):
        x1, y1 = point[0], point[1]
        total = 0.0
        for other_point in all_points:
            x2, y2 = other_point[0], other_point[1]
            if (x1, y1) != (x2, y2):
                dx = x1 - x2
                dy = y1 - y2
                total += math.sqrt(dx*dx + dy*dy)
        return total
    
    point_distances = [(point, calculate_total_distance(point, init_vehicle_pos_list)) 
                       for point in init_vehicle_pos_list]
    sorted_points = sorted(point_distances, key=lambda x: x[1])
    sorted_points = [point for point, _ in sorted_points]
    
    return sorted_points




def find_and_analyze_similar_roads(
    file_path,
    target_road,
    vehicle_info,
    target_speed=None,
    column_name='road',
    speed_column='speed',
    relation_column='relation',
    same_road_column='same_road',
    top_n=None,
    similarity_threshold=0.3,
    speed_tolerance=6.0
):
    try:
        df = pd.read_csv(file_path)
        
        required_cols = {column_name, relation_column, same_road_column, 'configuration'}
        if target_speed is not None:
            required_cols.add(speed_column)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"CSV missing required columns: {missing_cols}")

        df = df.dropna(subset=list(required_cols))
        df[column_name] = df[column_name].astype(str)
        df = df[df[column_name].apply(lambda x: all(c in VALID_ROAD_CHARS for c in x))]

        if target_speed is not None:
            df[speed_column] = pd.to_numeric(df[speed_column], errors='coerce')
            df = df.dropna(subset=[speed_column])
            df = df[abs(df[speed_column] - target_speed) <= speed_tolerance]

        df['similarity'] = df[column_name].apply(
            lambda x: 0.5 * calculate_road_similarity(target_road, x) + 
                     0.5 * get_position_weighted_similarity(target_road, x)
        )
        df = df[df['similarity'] >= similarity_threshold]

        if vehicle_info:
            cond = pd.Series(False, index=df.index)
            for rel, road in vehicle_info:
                cond |= (df[relation_column] == rel) & (df[same_road_column] == road)
            data_1 = df[cond].copy()
        else:
            data_1 = df.copy()

        config_counter = Counter(data_1['configuration'])
        configurations = [config for config, _ in config_counter.most_common()]
        configurations = configurations[:3]

        return data_1, configurations

    except Exception as e:
        print(f"Processing error: {str(e)}")
        return pd.DataFrame(), []

def find_three_logs(
    data_1,
    configuration,
    map_char=None,
    max_speed=None
):
    config_data = data_1[data_1['configuration'] == configuration].copy()
    
    if config_data.empty:
        return None
    
    complete_data = (
        config_data
        .sort_values('similarity', ascending=False)
        .head(3)
        .apply(
            lambda row: f"Danger vehicle is at {row['relation']} position of victim vehicle, same road flag is {row['same_road']} (0 means not same road, 1 means same road). Collision scenario description: {row['summary']}",
            axis=1
        )
        .tolist()
    )
    
    return complete_data

def call_llm_for_init_info(map_char, lane_num, max_speed, configuration, other_vehicle_info, danger_vehicle, three_logs):
    client = OpenAI(
        base_url='url',
        api_key='key'
    )

    start = time.time()
    
    prompt = f"""
    prompt
    """

    sys_prompt = """
    rules
    """
    response = client.chat.completions.create(
        model=model
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
    )

    content = response.choices[0].message.content

    end = time.time()
    print("LLM generated initialization results")
    print(f"Execution time: {end - start} seconds")
    return content

import simulate

def parse_llm_response(content, content_path):
    """
    Parse LLM response content to extract specific format tuple data
    
    Args:
        content: Raw response content from LLM
        configuration_path: Path to save configuration file
        
    Returns:
        List of parsed tuples, each in format (int, float, float, float, float, str)
    """
    with open(content_path, 'w', encoding='utf-8') as file:
        file.write(content)

    lines = content.split('\n')
    tuples = []
    cnt2 = 0
    
    for line in reversed(lines):
        if line.startswith('('):
            cnt2 += 1
            try:
                stripped = line[1:-1]
                parts = stripped.split(',', 5)
                if len(parts) == 6:
                    tuple_data = [
                        int(parts[0].strip()),
                        float(parts[1].strip()),
                        float(parts[2].strip()),
                        float(parts[3].strip()),
                        float(parts[4].strip()),
                        parts[5].strip()
                    ]
                    tuples.append(tuple(tuple_data))
            except (ValueError, IndexError) as e:
                print(f"Error processing line: {line}. Error: {e}")

        else:
            if cnt2 > 0:
                break

    tuples = tuples[::-1]
    return tuples

def llm_modifiy(iter_num, map_char, lane_num, max_speed, configuration, tuples, danger_car_trajectory, victim_car_trajectory, configuration_path):
    client = OpenAI(
            base_url='url',
            api_key='key'
        )

    start = time.time()

    sys_prompt = f"""
    rules
    """

    prompt = f"""
    prompt
    """
    
    response = client.chat.completions.create(
        model=model
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
    )

    content = response.choices[0].message.content

    print("LLM generated modification results")
    end = time.time()
    print(f"Execution time: {end - start} seconds")
    
    content_path = os.path.join(configuration_path, f"llm_response_{iter_num}.txt")
    tuples = parse_llm_response(content, content_path)
    simulate_info_path = os.path.join(configuration_path, f"tuples_{iter_num}.txt")
    with open(simulate_info_path, 'w') as f:
        for t in tuples:
            f.write(f"({t[0]}, {t[1]}, {t[2]}, {t[3]}, {t[4]}, {t[5]})\n")
    return tuples, simulate_info_path

def parse_line(line):
    line = line.strip()[1:-1]
    parts = []
    remaining = line
    for _ in range(5):
        split_pos = remaining.find(',')
        if split_pos == -1:
            break
        part = remaining[:split_pos].strip()
        parts.append(part)
        remaining = remaining[split_pos + 1:]
    parts.append(remaining.strip().strip('"'))
    return tuple(parts)

def extract_zero_first_items(filename):
    with open(filename, 'r') as file:
        for line in file:
            parsed = parse_line(line)
            if parsed[0] == '0':
                return float(parsed[1]), float(parsed[2])
    return None

def extract_one_first_items(filename):
    with open(filename, 'r') as file:
        for line in file:
            parsed = parse_line(line)
            if parsed[0] == '1':
                return float(parsed[1]), float(parsed[2])
    return None

def save_scenario_record(image_name, configuration_path, configuration, flag, danger_score, iter_num):
    print("saving scene record")
    scenario_filepath = SCENARIO_FILEPATH
    tuples_path = os.path.join(configuration_path, f"tuples_{iter_num}.txt")
    ego_car_x, ego_car_y = extract_zero_first_items(tuples_path)
    danger_car_x, danger_car_y = extract_one_first_items(tuples_path)
    
    with open(scenario_filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            image_name, 
            ego_car_x, 
            ego_car_y, 
            configuration, 
            tuples_path, 
            flag, 
            danger_score, 
            iter_num, 
            danger_car_x, 
            danger_car_y
        ])

csn_cnt = 0
cn_cnt = 0

for image_name in image_names: 
    print(f"current image_name: {image_name}")

    road_geo_file_path = os.path.join(road_geo_folder, f"{image_name}_road_geo.txt")
    map_char, lane_num, max_speed = parse_triplet(road_geo_file_path)
    init_vehicle_pos_list = []

    all_available_pos = file_read(os.path.join(img_scene_folder, f"{image_name}", "preparation", f"{image_name}_merge_checked.txt"))
    real_pos = file_read(os.path.join(img_pos_txt_folder, f"{image_name}.txt"))
    width, height = Image.open(os.path.join(img_pos_img_folder, f"{image_name}.png")).size
    llm_response_init_info_file = os.path.join(img_scene_folder, f"{image_name}", "preparation", f"{image_name}_llm_init_info.txt")
    init_vehilce_pos_file = os.path.join(img_scene_folder, f"{image_name}", "preparation", f"{image_name}_init_pos_list.txt")
    
    init_vehicle_pos_list = call_llm_for_init_pos(all_available_pos, real_pos, map_char, lane_num, max_speed, width, height, llm_response_init_info_file, init_vehilce_pos_file)

    input_file = init_vehilce_pos_file
    scene_generation_path = os.path.join(img_scene_folder, f"{image_name}", "scene_generation")
    if not os.path.exists(scene_generation_path):
        os.makedirs(scene_generation_path)
    output_file = os.path.join(img_scene_folder, f"{image_name}", "scene_generation", f"{image_name}_directions.json")
    vehicles = read_vehicle_data(input_file)
    structured_data = analyze_vehicles(vehicles)
    with open(output_file, 'w') as f:
        json.dump(structured_data, f, indent=2)
    print("saved_json")

    random_three_vehicles = random_choose_vehicles(init_vehicle_pos_list)
    searched_vehicle = 0
    success_case = 0
    for danger_vehicle in random_three_vehicles:    
        searched_vehicle = searched_vehicle + 1
        if success_case >= 6 or searched_vehicle >= 8:
            break
        else:
            danger_vehicle_x, danger_vehicle_y = danger_vehicle[0], danger_vehicle[1]
            print(f"danger_vehicle_x:{danger_vehicle_x}, danger_vehicle_y:{danger_vehicle_y}")
            
            vehicle_cnt_path = os.path.join(scene_generation_path, f"{danger_vehicle_x}_{danger_vehicle_y}")

            if not os.path.exists(vehicle_cnt_path):
                os.makedirs(vehicle_cnt_path)

            danger_vehicle_rel_pos_list = find_relations(structured_data, danger_vehicle)
            other_vehicles = find_other_vehicles(structured_data, danger_vehicle)
            other_vehicle_info = '\n'.join(str(tuple_item) for tuple_item in other_vehicles)

            data_1, configurations = find_and_analyze_similar_roads(
                file_path="path_to_crash_data.csv",
                target_road=map_char,
                vehicle_info=danger_vehicle_rel_pos_list,
                target_speed=max_speed,
                similarity_threshold=0.2,
                speed_tolerance=5.0
            )

            if not configurations:
                configuration_path = os.path.join(vehicle_cnt_path, "unknown")
                if not os.path.exists(configuration_path):
                    os.makedirs(configuration_path)
                three_logs = "No logs found"
                configuration = "unknown"
                content = call_llm_for_init_info(map_char, lane_num, max_speed, configuration, other_vehicles, danger_vehicle, three_logs)
                content_path = os.path.join(configuration_path, "llm_response_0.txt")
                
                tuples = parse_llm_response(content, content_path) 
                simulate_info_path = os.path.join(configuration_path, "tuples_0.txt")
                with open(simulate_info_path, 'w') as f:
                    for t in tuples:
                        f.write(f"({t[0]}, {t[1]}, {t[2]}, {t[3]}, {t[4]}, {t[5]})\n")

                filename = simulate_info_path        
                first_gif = os.path.join(configuration_path, f"first_frame_0.gif")
                video = os.path.join(configuration_path, f"video_0.gif")
                traj_path = os.path.join(configuration_path, f"traj_0.json")
                tm, tm_time, danger_car_trajectory, victim_car_trajectory, video_log, low_danger_score = simulate.simulator(filename, map_char, lane_num, first_gif, video, traj_path)
                low_score_iter = 0
                cn_cnt += 1
                if video_log[0]:
                    csn_cnt+=1
                    print(f"Successful collision {configuration} danger_vehicle_x:{danger_vehicle_x}, danger_vehicle_y:{danger_vehicle_y}")
                    save_scenario_record(image_name, configuration_path, configuration, 1, low_danger_score, 0)
                    success_case = success_case + 1
                else:
                    for iter_num in range(1, 6):
                        print(f"failed, itering number: {iter_num}" )
                        tuples, filename_modified = llm_modifiy(iter_num, map_char, lane_num, max_speed, configuration, tuples, danger_car_trajectory, victim_car_trajectory, configuration_path)
                        
                        first_gif = os.path.join(configuration_path, f"first_frame_{iter_num}.gif")
                        video = os.path.join(configuration_path, f"video_{iter_num}.gif")
                        traj_path = os.path.join(configuration_path, f"traj_{iter_num}.json")
                        tm, tm_time, danger_car_trajectory, victim_car_trajectory, video_log, danger_score = simulate.simulator(filename_modified, map_char, lane_num, first_gif, video, traj_path)
                        if video_log[0]:
                            print(f"Success after iteration {iter_num}")
                            print(f"Successful collision {configuration} danger_vehicle_x:{danger_vehicle_x}, danger_vehicle_y:{danger_vehicle_y}")
                            csn_cnt+=1
                            save_scenario_record(image_name, configuration_path, configuration, 1, danger_score, iter_num)
                            success_case = success_case + 1
                            break
                        else:
                            if danger_score < low_danger_score:
                                low_danger_score = danger_score
                                low_score_iter = iter_num

                            if iter_num < 5:
                                print("failed,continue to iter")
                            else:
                                print("failed to get any collisions")