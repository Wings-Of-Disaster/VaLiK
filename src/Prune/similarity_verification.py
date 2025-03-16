import argparse
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def calculate_similarity(image, texts):
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    outputs = model(**inputs)
    
    image_features = outputs.image_embeds
    text_features = outputs.text_embeds
    similarity = (image_features @ text_features.T).squeeze(0)
    return similarity.cpu().detach().numpy()

def chunk_text(text, mode="word", window_size=3):
    if mode == "word":
        return text.split()
    elif mode == "sentence":
        return sent_tokenize(text)
    elif mode == "window":
        words = text.split()
        return [' '.join(words[i:i+window_size]) for i in range(0, len(words), window_size)]
    else:
        raise ValueError("Invalid chunk mode")

def process_text(image_path, text, args):

    image = Image.open(image_path)
    
    chunks = chunk_text(text, mode=args.mode, window_size=args.window_size)
    if not chunks:
        return ""
    
    similarities = calculate_similarity(image, chunks)
    
    filtered_chunks = [chunk for chunk, sim in zip(chunks, similarities) if sim >= args.threshold]
    
    if args.mode == "word":
        return ' '.join(filtered_chunks)
    elif args.mode == "sentence":
        return ' '.join(filtered_chunks)
    elif args.mode == "window":
        return ' '.join(filtered_chunks)

def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def save_filtered_text(file_path, filtered_text):
    dir_path, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    
    new_filename = f"{name}_filtered{ext}"
    new_file_path = os.path.join(dir_path, new_filename)
    
    with open(new_file_path, 'w', encoding='utf-8') as file:
        file.write(filtered_text)
    
    print(f"Filtered text saved to: {new_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--text_path", type=str, required=True, help="Path to the text file containing the input text")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--mode", choices=["word", "sentence", "window"], default="word")

    #Please note that the maximum number of tokens is 77. Alternatively, you can also use other models for calculating similarity.
    parser.add_argument("--window_size", type=int, default=5)
    
    args = parser.parse_args()

    if args.mode == "sentence":
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

    text = read_text_from_file(args.text_path)
    result = process_text(args.image_path, text, args)
    print("Filtered Text:", result)

    save_filtered_text(args.text_path, result)