import os
from PIL import Image
from clip_interrogator import Config, Interrogator
import glob
import argparse

SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))

def process_single_image(image_file):
    base_name, _ = os.path.splitext(image_file)
    text_file = f"{base_name}.txt"
    if os.path.exists(text_file):
        print(f"Skipping {image_file}, description already exists.")
        return
    try:
        with Image.open(image_file).convert('RGB') as img:
            description = ci.interrogate(img)
            with open(text_file, 'w') as f:
                f.write(description)
            print(f"Description for {image_file} saved to {text_file}")
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

def process_batch_images(base_folder):
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTENSIONS):
                image_file = os.path.join(root, file)
                process_single_image(image_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images to generate descriptions.")
    parser.add_argument('--input', type=str, required=True, help="Path to a single image or a base folder containing images.")
    args = parser.parse_args()

    input_path = args.input
    if os.path.isfile(input_path) and input_path.lower().endswith(SUPPORTED_EXTENSIONS):
        process_single_image(input_path)
    elif os.path.isdir(input_path):
        process_batch_images(input_path)
    else:
        print(f"Invalid input path: {input_path}. Please provide a valid image file ({', '.join(SUPPORTED_EXTENSIONS)}) or a directory.")

    print("Processing completed!")