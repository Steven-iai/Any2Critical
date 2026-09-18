import os
import base64
import argparse
from openai import OpenAI

class RoadStructureExtractorBatch:
    def __init__(self, api_key: str, base_url: str = 'url', model: str = 'model'):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @staticmethod
    def encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def extract(self, image_path: str) -> str:
        base64_image = self.encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "prompt"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def process_image(self, image_path: str, output_dir: str):
        try:
            print(f"Processing image: {image_path}")
            result = self.extract(image_path)

            if result.startswith("(") and result.endswith(")"):
                parts = result[1:-1].split(",")
                if len(parts) == 3:
                    road_str = parts[0].strip()
                    lanes = int(parts[1].strip())
                    speed_kmh = float(parts[2].strip())
                    speed_ms = round(speed_kmh * 0.27, 2)
                    result_str = f"({road_str}, {lanes}, {speed_ms})"
                else:
                    result_str = "(INVALID_RESULT, 0, 0.0)"
            else:
                result_str = "(INVALID_RESULT, 0, 0.0)"

            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_road_geo.txt")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result_str)

            print(f"Result saved to {output_file}")
            return True
        except Exception as e:
            print(f"Failed to process {image_path}: {str(e)}")
            return False

    def process_directory(self, input_path: str, output_dir: str):
        if not os.path.exists(input_path):
            print(f"Path not found: {input_path}")
            return

        if os.path.isfile(input_path):
            self.process_image(input_path, output_dir)
            return

        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        processed_count = 0
        failed_count = 0

        for filename in os.listdir(input_path):
            file_path = os.path.join(input_path, filename)
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in supported_extensions:
                if self.process_image(file_path, output_dir):
                    processed_count += 1
                else:
                    failed_count += 1

        print(f"\nProcessing complete: {processed_count} succeeded, {failed_count} failed")

def main():
    parser = argparse.ArgumentParser(description="Batch extract road structure triples and convert speed limit to m/s")
    parser.add_argument("--input_path", default="input_images", help="Input image file or directory path")
    parser.add_argument("--output_dir", default="output_results", help="Output directory path")
    args = parser.parse_args()

    extractor = RoadStructureExtractorBatch(api_key='your-api-key')
    extractor.process_directory(args.input_path, args.output_dir)

if __name__ == "__main__":
    main()