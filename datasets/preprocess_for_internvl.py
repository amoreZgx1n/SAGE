import json
from PIL import Image
import os

def process_json(input_file, output_file, base_dir):
    # Read input JSON
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = []
    current_id = 0
    
    for item in data:
        # Get image path and dimensions
        image_path = item['images'][0]
        full_image_path = os.path.join(base_dir, image_path)
        
        try:
            with Image.open(full_image_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Error reading image {full_image_path}: {e}")
            continue
            
        # Process each QA pair
        messages = item['messages']
        for i in range(0, len(messages), 2):  # Process pairs of messages
            if i + 1 >= len(messages):
                break
                
            entry = {
                "id": current_id,
                "image": image_path,
                "width": width,
                "height": height,
                "conversations": [
                    {
                        "from": "human",
                        "value": messages[i]['content'] if '<image>' in messages[i]['content'] 
                                else '<image>' + messages[i]['content']
                    },
                    {
                        "from": "gpt",
                        "value": messages[i + 1]['content']
                    }
                ]
            }
            
            output_data.append(entry)
            current_id += 1
    
    # Write output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    base_dir = "workplace/internVL2"
    input_file = "/workplace/internVL2/dataset/data_process/latest_data/test.json"
    output_file = "workplace/internVL2/dataset/internvl_format_latest/internvl2_test.json"
    
    process_json(input_file, output_file, base_dir)