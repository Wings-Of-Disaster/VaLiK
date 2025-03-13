import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import argparse
import os
import glob
import deepspeed
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import numpy as np
import cv2
import base64

import time

def load_image(image_path):
    raw_image = Image.open(image_path).convert('RGB')
    return raw_image

def image_to_base64(image):
    image_np = np.array(image)
    if image_np.shape[0] in [3, 4]:
        image_np = np.transpose(image_np, (1, 2, 0))
    if image_np.dtype != np.uint8:
        image_np = (255 * image_np).astype(np.uint8)
    if image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    _, buffer = cv2.imencode('.jpg', image_np)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

def generate_visual_knowledge(image, model, processor, device, prompt):
    base64_image = image_to_base64(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": f"data:image;base64,{base64_image}"},
            ],
        }
    ]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

        #generation_args = {
        #"max_new_tokens": 300,
        #"num_beams": 3,
        #"do_sample": True,
        #"temperature": 0.7,
        #"top_p": 0.9,
        #"no_repeat_ngram_size": 3}

    #generated_ids = model.generate(**inputs, **generation_args)
    generated_ids = model.generate(**inputs, max_new_tokens=32768)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    description = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return description.strip()

def process_single_image(image_path, model, processor, device, prompt, output_txt_path=None):
    try:
        input_image = load_image(image_path)
        print(f"Analyzing image: {image_path}...\n")
        description = generate_visual_knowledge(input_image, model, processor, device, prompt)
        print("[Description]\n", description, "\n")

        if output_txt_path:
            with open(output_txt_path, 'w') as f:
                f.write(f"[Description]\n{description}\n")
            print(f"Description saved to {output_txt_path}")
        
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")

def process_batch_images(folder_path, model, processor, device, prompt):
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
            process_single_image(image_path, model, processor, device, prompt, output_txt_path=txt_path)

def load_model_with_quantization(model_name, use_quantization):
    if use_quantization:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    else:
        quantization_config = None

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
        trust_remote_code=True
    )

    return processor, model

def load_model_with_deepspeed(model_name):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model_engine = deepspeed.init_inference(
        model=model,
        mp_size=torch.cuda.device_count(),
        dtype=torch.float16,
        replace_method='auto',
        replace_with_kernel_inject=True,
        #max_tokens=1500,
        #model_parameters=model.parameters(),
        #config=deepspeed_config
    )
    #assert isinstance(model_engine.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"
    return processor, model_engine

def load_model_standard(model_name):
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name
    )
    return processor, model

if __name__ == '__main__':

    #Please use this command to execute deepseed, but we recommend quantization.
    #deepspeed --num_gpus 4 Qwen2VL_ScienceQA.py --input your_image --model qwen2-vl-instruct:72b --use_deepspeed
    
    parser = argparse.ArgumentParser(description="Generate visual descriptions for images using Qwen2-VL-Instruct.")
    parser.add_argument('--input', type=str, required=True, help="Path to a single image or a folder of images.")
    parser.add_argument('--model', type=str, required=True, choices=['qwen2-vl-instruct:2b', 'qwen2-vl-instruct:7b', 'qwen2-vl-instruct:72b'], help="Model size: '7b', '14b', or '72b'.")
    parser.add_argument('--use_quantization', action='store_true', help="Use 4-bit quantization.")
    parser.add_argument('--use_deepspeed', action='store_true', help="Use DeepSpeed for distributed training.")
    #parser.add_argument('--deepspeed_config', type=str, default="ds_config.json", help="Path to DeepSpeed config file.")
    parser.add_argument('--local_rank', type=int, default=-1, help="DeepSpeed internal use (auto-injected).")
    args = parser.parse_args()

    model_map = {
        'qwen2-vl-instruct:2b': "Qwen/Qwen2-VL-2B-Instruct",
        'qwen2-vl-instruct:7b': "Qwen/Qwen2-VL-7B-Instruct",
        'qwen2-vl-instruct:72b': "Qwen/Qwen2-VL-72B-Instruct"
    }
    model_name = model_map[args.model]

    prompt = """
    Please provide a detailed visual description of this image. 
    Include key objects, their spatial relationships, 
    notable visual features, and any observable actions or events.
    Respond in clear, structured English paragraphs.
    """

    if args.use_deepspeed:
        processor, model = load_model_with_deepspeed(model_name)
        device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cuda")
    elif args.use_quantization:
        processor, model = load_model_with_quantization(model_name, args.use_quantization)
        device = "cuda"
    else:
        processor, model = load_model_standard(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

    if os.path.isfile(args.input):
        output_txt_path = os.path.splitext(args.input)[0] + '.txt'
        process_single_image(args.input, model, processor, device, prompt, output_txt_path=output_txt_path)
    elif os.path.isdir(args.input):
        process_batch_images(args.input, model, processor, device, prompt)
    else:
        print(f"Invalid input path: {args.input}")