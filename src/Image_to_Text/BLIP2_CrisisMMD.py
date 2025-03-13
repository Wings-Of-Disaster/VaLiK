import torch
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration

import argparse
import os

def load_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    return raw_image

def generate_visual_knowledge(image, model, processor, device):

    #prompt = """
    #Please provide a detailed visual description of this image. 
    #Include key objects, their spatial relationships, 
    #notable visual features, and any observable actions or events.
    #Respond in clear, structured English paragraphs.
    #"""

    prompt = """
    Please provide a detailed visual description of this image. 
    """

    inputs = processor(
        images=image, 
        text=prompt, 
        return_tensors="pt"
    ).to(device)

    generation_args = {
        "max_length": 300,
        "num_beams": 5,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "no_repeat_ngram_size": 3
    }

    generated_ids = model.generate(**inputs, **generation_args)
    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return description.strip()

def process_single_image(image_path, model, processor, device, output_txt_path=None):

    try:
        input_image = load_image(image_path)
        print(f"Analyzing image: {image_path}...\n")
        description = generate_visual_knowledge(input_image, model, processor, device)
        print("[Description]\n", description, "\n")
        
        if output_txt_path:
            with open(output_txt_path, 'w') as f:
                f.write(f"[Description]\n{description}\n")
            print(f"Description saved to {output_txt_path}")
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

def process_batch_images(folder_path, model, processor, device):

    supported_extensions = ['.jpg', '.jpeg', '.png']
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in supported_extensions):
                image_path = os.path.join(root, file)
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                
                if os.path.exists(txt_path):
                    print(f"Skipping {image_path}, description already exists.")
                    continue
                process_single_image(image_path, model, processor, device, output_txt_path=txt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate visual descriptions for images.")
    parser.add_argument('--input', type=str, required=True, help="Path to a single image or a folder of images.")
    parser.add_argument('--model_type', type=str, required=True, choices=['flan-t5', 'opt'], help="Model type: 'flan-t5' or 'opt'.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.model_type == 'flan-t5':
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    elif args.model_type == 'opt':
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    model.to(device)
    
    if os.path.isfile(args.input):
        output_txt_path = os.path.splitext(args.input)[0] + '.txt'
        process_single_image(args.input, model, processor, device, output_txt_path=output_txt_path)
    elif os.path.isdir(args.input):
        process_batch_images(args.input, model, processor, device)
    else:
        print(f"Invalid input path: {args.input}")