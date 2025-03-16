# Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning

This is an open-source implementation of VaLiK framework for enhanced LLM reasoning.

![Figure 1: VaLiK Pipeline](pipeline.jpg)

## Install

You can create a Conda environment and install dependencies using requirements.txt :
```bash
conda create --name valik python=3.10
conda activate valik
pip install -r requirements.txt
```
Or setup environment with provided YML :
`conda env create -f environment.yml`

## Usage

### CoE-based Visual to Language Modeling
We provide examples and detailed instructions in the `src/Image_to_Text` folder. You can use this module to generate descriptions for a single image or all images in a folder. The descriptions will be saved in corresponding `.txt` files with the same name as the images.
If you want to test with the dataset we used, you can run
`bash datasets/Preprocess_ScienceQA.sh`
to preprocess the data.

### Cross-Modal Similarity Verification
Pruning is a double-edged sword. Whether to employ pruning techniques and what thresholds to set depend on the specific requirements of the task. Different VLMs generate sentences in various forms. We've provided relevant resources under the `src/Prune` directory for reference.
To conduct the verification, run the following command:
`python src/Prune/similarity_verification.py --image_path datasets/ScienceQA/data/scienceqa/images/train/1/image.png --text_path datasets/ScienceQA/data/scienceqa/images/train/1/image.txt --threshold 0.20 --mode sentence`

### MMKG Construction for Enhanced Reasoning
We utilize LightRAG, a lightweight framework to construct MMKGs. For comprehensive details regarding LightRAG, kindly visit the official repository: [https://github.com/HKUDS/LightRAG](https://github.com/HKUDS/LightRAG).
To construct the multimodal knowledge graph, run the following command:
```bash
python src/LightRAG/lightrag_ollama_demo.py
```
Note: Different LLMs can impact the graph construction time. We recommend using the Qwen2.5 model for graph construction as it strikes a good balance between efficiency and effectiveness.