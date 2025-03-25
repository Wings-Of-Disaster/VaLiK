import os
import logging
import argparse
import glob
import base64
import torch
import requests
from io import BytesIO
from PIL import Image
from qwen_vl_utils import process_vision_info
from clip_interrogator import Config, Interrogator
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    AutoModelForImageTextToText,
    BitsAndBytesConfig
)
try:
    import ollama
except ImportError:
    # Allow environments without ollama installed
    pass  

class VisualDescriptionGenerator:
    def __init__(self, args):
        self.args = args
        self.device = self._get_device()
        self.prompt = """
        Please provide a detailed visual description of this image. 
        Include key objects, their spatial relationships, 
        notable visual features, and any observable actions or events.
        Respond in clear, structured English paragraphs.
        """
        
        if self.args.model_type == 'api':
            self._validate_api_credentials()
        elif self.args.model_type == 'blip2':
            self.processor, self.model = self._init_BLIP2()
        elif self.args.model_type == 'llava':
            self.client = self._init_LLaVa()
        elif self.args.model_type == 'qwen2-vl':
            self.processor, self.model = self._init_Qwen2_VL()
        elif self.args.model_type == 'clip-interrogator':
            self.ci = self._init_CLIP_Interrogator()

    def _get_device(self):

        if self.args.model_type in ['api', 'llava']:
            return None
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _validate_api_credentials(self):

        if not hasattr(self.args, 'api_key') or not self.args.api_key:
            raise ValueError("API key is required.")

    def _init_BLIP2(self):

        model_map = {
            'flan-t5': "Salesforce/blip2-flan-t5-xl",
            'opt': "Salesforce/blip2-opt-2.7b"
        }
        processor = AutoProcessor.from_pretrained(model_map[self.args.blip2_version])
        model = Blip2ForConditionalGeneration.from_pretrained(
            model_map[self.args.blip2_version]
        ).to(self.device)
        return processor, model

    def _init_LLaVa(self):

        return ollama.Client(host=f"http://localhost:{self.args.llava_port}")

    def _init_Qwen2_VL(self):

        model_map = {
            '2b': "Qwen/Qwen2-VL-2B-Instruct",
            '7b': "Qwen/Qwen2-VL-7B-Instruct",
            '72b': "Qwen/Qwen2-VL-72B-Instruct"
        }

        if self.args.use_quantization:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4"
            )
        else:
            quant_config = None

        processor = AutoProcessor.from_pretrained(
            model_map[self.args.qwen2vl_version],
            trust_remote_code=True
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_map[self.args.qwen2vl_version],
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            trust_remote_code=True
        ).to(self.device)
        return processor, model
    
    def _init_CLIP_Interrogator(self):
        config = Config()
        if hasattr(self.args, 'clip_model'):
            config.clip_model_name = self.args.clip_model
        return Interrogator(config)

    def generate_description(self, image_path):

        if self.args.model_type == 'api':
            return self._generate_API_description(image_path)
        elif self.args.model_type == 'blip2':
            return self._generate_BLIP2_description(image_path)
        elif self.args.model_type == 'llava':
            return self._generate_LLaVa_description(image_path)
        elif self.args.model_type == 'qwen2-vl':
            return self._generate_Qwen2_VL_description(image_path)
        elif self.args.model_type == 'clip-interrogator':
            return self._generate_CLIP_Interrogator_description(image_path)

    def _generate_API_description(self, image_path):

        base64_image = self._image_to_base64(image_path)
        headers = {"Authorization": f"Bearer {self.args.api_key}"}
        data = {
            "model": "gpt-4o",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }]
        }
        response = requests.post(self.args.api_url, headers=headers, json=data)
        return response.json()['choices'][0]['message']['content'].strip()

    def _generate_BLIP2_description(self, image_path):

        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(
            images=image,
            text=self.prompt,
            return_tensors="pt"
        ).to(self.device)
        
        generated_ids = self.model.generate(
            **inputs,
            max_length=300,
            num_beams=5,
            temperature=0.7
        )
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)

    def _generate_LLaVa_description(self, image_path):

        base64_image = self._image_to_base64(image_path)
        response = self.client.chat(
            model=f"llava:{self.args.llava_version}",
            messages=[{
                'role': 'user',
                'content': self.prompt,
                'images': [base64_image]
            }]
        )
        return response['message']['content'].strip()

    def _generate_Qwen2_VL_description(self, image_path):

        base64_image = self._image_to_base64(image_path)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt},
                {"type": "image", "image": f"data:image;base64,{base64_image}"}
            ]
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=1500,
            temperature=0.7
        )
        return self.processor.decode(generated_ids[0], skip_special_tokens=True)

    def _generate_CLIP_Interrogator_description(self, image_path):
        try:
            with Image.open(image_path).convert('RGB') as img:
                return self.ci.interrogate(img)
        except Exception as e:
            print(f"CLIP processing error: {str(e)}")
            return None

    @staticmethod
    def _image_to_base64(image_path):

        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.convert("RGB").save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def process(self):

        if os.path.isfile(self.args.input):
            self._process_single_image(self.args.input)
        elif os.path.isdir(self.args.input):
            self._process_batch_images(self.args.input)
        else:
            raise ValueError(f"Invalid input path: {self.args.input}")

    def _process_single_image(self, image_path):

        try:
            print(f"\nAnalyzing image: {image_path}")
            description = self.generate_description(image_path)
            print(f"\n[Description]\n{description}")
            
            output_path = os.path.splitext(image_path)[0] + '.txt'
            with open(output_path, 'w') as f:
                f.write(f"[Description]\n{description}")
            print(f"Saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    def _process_batch_images(self, folder_path):
        
        extensions = ['*.jpg', '*.jpeg', '*.png']
        for ext in extensions:
            for image_path in glob.glob(os.path.join(folder_path, '**', ext), recursive=True):
                txt_path = os.path.splitext(image_path)[0] + '.txt'
                if not os.path.exists(txt_path):
                    self._process_single_image(image_path)
                else:
                    print('Text file already exists.')
                    

def main():
    parser = argparse.ArgumentParser(description="Multimodel visual description generation tool")
    parser.add_argument('--input', required=True, help="The image path or folder you need to process")
    
    subparsers = parser.add_subparsers(dest='model_type', required=True)
    
    api_parser = subparsers.add_parser('api')
    api_parser.add_argument('--api_key', required=True, help="API-key")
    api_parser.add_argument('--api_url', default="https://api.openai.com/v1/chat/completions")
    
    blip_parser = subparsers.add_parser('blip2')
    blip_parser.add_argument('--blip2_version', choices=['flan-t5', 'opt'], required=True)
    
    llava_parser = subparsers.add_parser('llava')
    llava_parser.add_argument('--llava_version', choices=['7b', '13b', '34b'], required=True)
    llava_parser.add_argument('--llava_port', type=int, default=11434)
    
    qwen_parser = subparsers.add_parser('qwen2-vl')
    qwen_parser.add_argument('--qwen2vl_version', choices=['2b', '7b', '72b'], required=True)
    qwen_parser.add_argument('--use_quantization', action='store_true')

    clip_parser = subparsers.add_parser('clip-interrogator')
    clip_parser.add_argument('--clip_model', default="ViT-L-14/openai", help="CLIP model type")

    args = parser.parse_args()
    
    try:
        generator = VisualDescriptionGenerator(args)
        generator.process()
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()