import os
from PIL import Image
from clip_interrogator import Config, Interrogator
import torch
from torchvision import transforms
import glob

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

def process_batch_images(base_folder):
    sub_folders = ['train', 'val', 'test']
    for sub_folder in sub_folders:
        sub_folder_path = os.path.join(base_folder, sub_folder)
        if os.path.isdir(sub_folder_path):
            image_files = glob.glob(os.path.join(sub_folder_path, '**', '*.jpg'), recursive=True) + \
                          glob.glob(os.path.join(sub_folder_path, '**', '*.png'), recursive=True)
            for image_file in image_files:
                text_file = os.path.splitext(image_file)[0] + '.txt'
                if os.path.exists(text_file):
                    print(f"Skipping {image_file}, description already exists.")
                    continue
                try:
                    with Image.open(image_file).convert('RGB') as img:
                        description = ci.interrogate(img)
                        with open(text_file, 'w') as f:
                            f.write(description)
                        print(f"Description for {image_file} saved to {text_file}")
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")

base_folder = '../../datasets/ScienceQA/data/scienceqa/images'
process_batch_images(base_folder)

print("Processing completed!")