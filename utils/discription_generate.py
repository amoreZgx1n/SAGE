import base64
import os

import json

from openai import OpenAI
from tqdm import tqdm

# 设置API密钥
os.environ["OPENAI_API_KEY"] = "${api-key}"
os.environ["OPENAI_BASE_URL"] = "https://api.bltcy.ai/v1"
openai_client = OpenAI()

prompt = '''
Analyze the given industrial image and generate a **detailed** 5-10 sentence description, covering the following aspects:

1. **Object Description**: Describe the industrial object in the image, including its material, structure, and function.
2. **Potential Defects**: Identify possible defects that may occur in real-world conditions, such as cracks, corrosion, deformation, wear, or misalignment.
3. **Impact of Defects**: Explain how these defects could affect the performance, safety, lifespan, or operational efficiency of the object.

Use **precise industrial terminology** and provide a **professional assessment** of the potential risks and consequences of these defects.

Example 1: Metal Pipe
Input: Image of a metal pipeline
Output:
1. The image shows an industrial metal pipeline, typically made of stainless steel or carbon steel, commonly used in the chemical, oil, and gas industries.  

2. The primary function of this pipeline is to transport high-pressure or high-temperature fluids while maintaining structural integrity under extreme conditions.  

3. Over time, the pipeline may develop **corrosion**, especially when exposed to moisture, chemicals, or high humidity.  

4. Another common issue is **fatigue cracks**, which can form due to prolonged exposure to cyclic pressure and mechanical stress.  

5. If the pipeline is improperly welded, **weld defects** such as porosity or incomplete fusion may compromise its strength.  

6. Corrosion and cracks can lead to **leakage**, causing pressure loss and potential contamination of transported substances.  

7. In critical applications, a pipeline failure could result in **hazardous material spills, environmental damage, or operational downtime**.  

8. Regular **ultrasonic inspections and protective coatings** are recommended to prevent corrosion-related failures.  

9. If cracks are detected early, **weld repairs or reinforcement techniques** can be applied to extend the pipeline’s service life.  
'''

def get_disease_visual_characteristics(model, histories, max_tokens):
    if model == 'deepseek-coder':
        temperature = 1.0
    else:
        temperature = 0.0
    if max_tokens:
        answer = openai_client.chat.completions.create(
                    model=model,
                    messages=histories,
                    max_tokens = max_tokens,
                    temperature = temperature
                ).choices[0].message.content
    else:
        answer = openai_client.chat.completions.create(
                    model=model,
                    messages=histories,
                    temperature = temperature
                ).choices[0].message.content
    return answer

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

def generate_description(image_path):
    base_path = "/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2"
    image_path = os.path.join(base_path, image_path)
    content = [
        {
            "type": "text",
            "text": prompt
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image(image_path)}"
            }
        }
    ]
    histories = [{"role": "user", "content": content}]
    response_text = get_disease_visual_characteristics("gpt-4o", histories, max_tokens=3000)
    return response_text

if __name__ == '__main__':
    res = generate_description("dataset/MANTA_SETTING2_DATASET/ImageData/agriculture/maize/test/mold/agriculture_maize_10059.png")
    print(res)