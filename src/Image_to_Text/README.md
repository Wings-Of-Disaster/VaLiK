# Visual Language Models (VLMs) for Image Captioning

This repository provides examples of generating textual descriptions from images using different Visual Language Models (VLMs). The supported models include **CLIP-Interrogator**, **BLIP2**, **LLaVA**, and **Qwen2-VL**.

<img src="./assets/example.png" width="800">

Files with dataset suffixes (e.g., `BLIP2_CrisisMMD.py`) contain specialized processing pipelines for particular datasets, while files without suffixes handle general image processing through the `--input` argument. Below we outline each model's distinctive characteristics and deployment requirements.

---

## Supported Models

### 1. **CLIP-Interrogator**
- **Source**: [CLIP-Interrogator Repository](https://github.com/pharmapsychotic/clip-interrogator)
- **Supported VLMs**: CLIP, BLIP, and BLIP2.
- **Note**: The generated descriptions may include prompt-like text optimized for Stable Diffusion, which can introduce noise in some cases.

#### Usage
To run the CLIP-Interrogator on CrisisMMD or ScienceQA dataset, use the following command:
```bash
python CLIP_Interrogator_CrisisMMD.py
```
This can directly process our datasets, or you can also modify the file to change the path.

---

### 2. **BLIP2**
- **Implementation**: Uses the Hugging Face `transformers` library.
- **Supported LLMs**: OPT and FlanT5-XL.
- **Evaluation**: FlanT5-XL performs better but tends to generate shorter descriptions, making it suitable only as a first-layer VLM.

#### Usage
To run the BLIP-2 on your dataset, use the following command:
```bash
python BLIP2_CrisisMMD.py --input ../../datasets/CrisisMMD_v2.0/data_image --model_type flan-t5
```
The `input` parameter supports either a single image or a folder. This functionality also applies to the input parameter in subsequent commands.

---

### 3. **LLaVA**
- **Deployment**: We use **Ollama** for easy deployment.
- **Performance**: LLaVA is one of the best-performing VLMs.
- **Flexibility**: LLaVA supports both base64-encoded image data and direct file paths for image input. In our implementation, we demonstrate passing images as base64-encoded strings, which provides a programmatic and flexible way to integrate with Python scripts.
- **Recommendation**: Use LLaVA for high-quality descriptions when direct file paths are feasible.

#### Usage
To run the LLaVA on your dataset, use the following command:
```bash
python LLaVA_CrisisMMD.py --input ../../datasets/CrisisMMD_v2.0/data_image --model llava:7b --port 11434
```
Here, `llava:7b` is a parameter for specifying the model, and `port` is the port number for Ollama.

---

### 4. **Qwen2-VL**
- **Version**: We use the **Qwen2-VL-Instruct** version.
- **Implementation**: Loaded via the Hugging Face `transformers` library. Alternatively, you can use frameworks like **vLLM** for deployment.
- **Hardware Requirements**:
  - **2B and 7B Models**: Can be run on a single A100-80GB GPU.
  - **72B Model**: Requires significant resources. We provide two deployment options:
    1. **Quantization**: Recommended for single-GPU deployment. According to the [evaluation report](https://modelscope.cn/models/Qwen/Qwen2.5-VL-72B-Instruct), quantization does not significantly degrade performance and offers faster inference speeds.
    2. **DeepSpeed**: Suitable for distributed setups.
- **Performance**: Qwen2-VL is one of the best-performing VLMs, especially for detailed and structured descriptions.

#### Usage
To run the Qwen2-VL-Instruct on your dataset, you can choose different ways according to your needs.
If you want to use quantization, use the following command:
```bash
python Qwen2VL_CrisisMMD.py --input ../../datasets/CrisisMMD_v2.0/data_image --model qwen2-vl-instruct:72b --use_quantization
```
Here, using `--use_quantization` enables quantization for the model.
If you want to use DeepSpeed for distributed inference, use the following command:
```bash
deepspeed --num_gpus 8 Qwen2VL_CrisisMMD.py --input ../../datasets/CrisisMMD_v2.0/data_image --model qwen2-vl-instruct:72b --use_deepspeed
```
Using `--use_deepspeed` enables DeepSpeed distributed inference, and `--num_gpus 8` specifies that 8 GPUs will be used for this operation.

---

## API Example
We provide a sample API script for remote inference. For example, you can integrate VLMs like Gemini and GPT-4o into your applications for remote calls.

#### Usage
To call the API for processing your dataset, use the following command:
```bash
python API_CrisisMMD.py --input ../../datasets/CrisisMMD_v2.0/data_image
```

---

## Additional Notes
- You can experiment with other VLMs and frameworks for image captioning. For deployment and performance tuning, refer to the official documentation of each model and framework.
- For subsequent VLMs in the CoE pipeline, the prompts need to be modified to inform them of the previous description and ask them to add new visual details.