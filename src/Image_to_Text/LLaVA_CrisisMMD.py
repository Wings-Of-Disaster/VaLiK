import os
import logging
import argparse
import glob
import ollama
import base64
from io import BytesIO
from PIL import Image

def convert_to_base64(image_path):
    try:
        with Image.open(image_path) as pil_image:
            buffered = BytesIO()
            pil_image.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def generate_visual_knowledge(image_path, prompt):
    try:
        base64_image = convert_to_base64(image_path)
        if not base64_image:
            return None

        response = client.chat(
            model=args.model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [base64_image]
            }]
        )
        return response['message']['content'].strip()

    except Exception as e:
        logging.error(f"API request failed: {str(e)}")
        return None

def process_single_image(image_path, prompt, output_txt_path=None):
    try:
        print(f"Analyzing image: {image_path}...\n")
        description = generate_visual_knowledge(image_path, prompt)
        
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
    supported_extensions = ['.jpg', '.jpeg', '.png']
    for root, _, files in os.walk(folder_path):
        image_files = []
        for ext in supported_extensions:
            image_files.extend(glob.glob(os.path.join(root, f'*{ext}')))
        print(f"Found {len(image_files)} images in {root}")
        
        for image_path in image_files:
            txt_path = os.path.splitext(image_path)[0] + '.txt'
            if os.path.exists(txt_path):
                print(f"Skipping {image_path}, description already exists.")
                continue
            process_single_image(image_path, prompt, output_txt_path=txt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visual descriptions using LLaVA via Ollama.")
    parser.add_argument('--input', type=str, required=True, help="Path to an image or image folder.")
    parser.add_argument('--model', type=str, required=True, choices=['llava:7b', 'llava:13b', 'llava:34b'], 
                       help="LLaVA model version")
    parser.add_argument('--port', type=int, default=11434,
                       help="Ollama service port (default: %(default)d)")
    args = parser.parse_args()

    # Initialize Ollama client
    client = ollama.Client(host=f"http://localhost:{args.port}")

    prompt = """Please provide a detailed visual description of this image. 
    Include key objects, their spatial relationships, 
    notable visual features, and any observable actions or events.
    Respond in clear, structured English paragraphs."""

    if os.path.isfile(args.input):
        output_txt_path = os.path.splitext(args.input)[0] + '.txt'
        process_single_image(args.input, prompt, output_txt_path=output_txt_path)
    elif os.path.isdir(args.input):
        process_batch_images(args.input, prompt)
    else:
        print(f"Invalid input path: {args.input}")