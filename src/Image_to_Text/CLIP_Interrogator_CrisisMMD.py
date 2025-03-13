import os
from PIL import Image
from clip_interrogator import Config, Interrogator
import torch
from torchvision import transforms
import glob

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

def process_batch_images(base_folder):
    for category_folder in os.listdir(base_folder):
        category_path = os.path.join(base_folder, category_folder)
        if os.path.isdir(category_path):
            for date_folder in os.listdir(category_path):
                date_path = os.path.join(category_path, date_folder)
                if os.path.isdir(date_path):
                    image_files = glob.glob(os.path.join(date_path, "*.jpg"))
                    for image_file in image_files:
                        text_file = image_file.replace('.jpg', '.txt')
                        # Check if the corresponding .txt file already exists
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

base_folder = '../../datasets/CrisisMMD_v2.0/data_image'

process_batch_images(base_folder)

print("Processing completed!")
