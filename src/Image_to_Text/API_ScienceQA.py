import os
import logging
import requests
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import re
import json
import base64
import io
from PIL import Image
import argparse
import glob

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_visual_knowledge(image, prompt):
    base64_image = image_to_base64(image)
    rl = "your-url"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-your-key"
    }
    data = {
        "model": "gpt-4o",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }]
    }
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        response_json = response.json()
        return response_json['choices'][0]['message']['content'].strip()
    else:
        logging.error(f"API request failed with status code: {response.status_code}")
        return None

def process_single_image(image_path, prompt, output_txt_path=None):
    try:
        input_image = load_image(image_path)
        print(f"Analyzing image: {image_path}...\n")
        description = generate_visual_knowledge(input_image, prompt)
        
        if description:
            print("[Description]\n", description, "\n")
            if output_txt_path:
                with open(output_txt_path, 'w') as f:
                    f.write(f"[Description]\n{description}\n")
                print(f"Description saved to {output_txt_path}")
        else:
            print(f"Failed to generate description for {image_path}")
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

def process_batch_images(folder_path, prompt):
    sub_folders = ['train', 'val', 'test']
    supported_extensions = ['.jpg', '.jpeg', '.png']
    
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if not os.path.isdir(sub_folder_path):
            print(f"Skipping {sub_folder_path}, not a directory.")
            continue
        
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(sub_folder_path, '**', f'*{ext}'), recursive=True))
        print(f"Found {len(image_files)} images in {sub_folder_path}")
        
        for image_path in image_files:
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(txt_path):
                print(f"Skipping {image_path}, description already exists.")
                continue
            process_single_image(image_path, prompt, output_txt_path=txt_path)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Generate visual descriptions for images using GPT-4o API.")
    parser.add_argument('--input', type=str, required=True, help="Path to a single image or a folder of images.")
    args = parser.parse_args()

    prompt = """
    Please provide a detailed visual description of this image. 
    Include key objects, their spatial relationships, 
    notable visual features, and any observable actions or events.
    Respond in clear, structured English paragraphs.
    """

    if os.path.isfile(args.input):
        output_txt_path = os.path.splitext(args.input)[0] + '.txt'
        process_single_image(args.input, prompt, output_txt_path=output_txt_path)
    elif os.path.isdir(args.input):
        process_batch_images(args.input, prompt)
    else:
        print(f"Invalid input path: {args.input}")