import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

import json

WORKING_DIR = "./ScienceQA_Text_and_Image_R1"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="deepseek-r1:72b",
    llm_model_max_async=160,
    llm_model_max_token_size=65536,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 65536}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

directory = '../../datasets/ScienceQA/data/scienceqa/images/train'
original_text_path = '../Original_Text_Compilation/ScienceQA_Text.txt'

try:
    with open(original_text_path, 'r', encoding='utf-8') as original_file:
        result_str = original_file.read()
except FileNotFoundError:
    print(f"Error: The file {original_text_path} does not exist.")
    result_str = ""
except Exception as e:
    print(f"Error reading {original_text_path}: {e}")
    result_str = ""

caption_string = ""
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('.txt'):
            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                caption_string += text

combine_string = result_str + caption_string

rag.insert(combine_string)

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)

# stream response
resp = rag.query(
    "What are the top themes in this story?",
    param=QueryParam(mode="hybrid", stream=True),
)


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)
