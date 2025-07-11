import json
import os
import numpy as np
import torch
from tqdm import tqdm
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def is_positive_answer(answer):
    answer = answer.lower().strip()
    positive_starts = ['true', 'true.', 'yes', 'yes.']
    return any(answer.startswith(start) for start in positive_starts)

def load_qa_files(qa_files):
    """Load and combine all QA JSON files into a single list"""
    qa_data = []
    for file in qa_files:
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            qa_data.extend(data)
    return qa_data

def pair_match(vanilla_data, image):
    """Find the QA pair that matches the given image"""
    for item in vanilla_data:
        if os.path.basename(item['test_image']) == os.path.basename(image):
            return item["pair_type"]

def test(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    output_data = []

    path = 'internvl2_8b'

    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

    vanilla_files = [
    '/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/dataset/MANTA_SETTING2_DATASET/JsonData/electronics_QA.json',
    '/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/dataset/MANTA_SETTING2_DATASET/JsonData/agriculture_QA.json',
    '/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/dataset/MANTA_SETTING2_DATASET/JsonData/groceries_QA.json',
    '/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/dataset/MANTA_SETTING2_DATASET/JsonData/mechanics_QA.json',
    '/mnt/pfs-mc0p4k/tts/team/zgx/workplace/internVL2/dataset/MANTA_SETTING2_DATASET/JsonData/medicine_QA.json'
    ]
    vanilla_data = load_qa_files(vanilla_files)

    for item in tqdm(data, desc="Processing QA pairs", unit="item"):
        question = item['conversations'][0]['value']
        answer = item['conversations'][1]['value']
        label = pair_match(vanilla_data, item['image'])
        
        is_true = is_positive_answer(answer)
        
        new_question = question + " Answer with True or False and provide a reason."
        pixel_values = load_image(item['image'], max_num=12).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        
        response = model.chat(tokenizer, pixel_values, new_question, generation_config)
        dichotomy = is_positive_answer(response)
        
        output_item = {
            "id": item['id'],
            "image": item['image'],
            "label": label,
            "question": question,
            "answer": answer,
            "prediction": response,
            "ground_truth": is_true,
            "dichotomy": dichotomy
        }
        output_data.append(output_item)
        print(response)
        print(dichotomy)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_file = "dataset/internvl_format_latest/internvl2_train.json"
    output_file = "result/internvl2/2_1ssft_v3.json"
    
    test(input_file, output_file)