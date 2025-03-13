# Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning

This is an open-source implementation of VaLiK framework for enhanced LLM reasoning.

![Figure 1: VaLiK Pipeline](pipeline.jpg)

## Install
Install dependencies using requirements.txt  
`pip install -r requirements.txt`  

Or setup environment with provided YML  
`conda env create -f environment.yml`

## Usage

### CoE-based Visual to Language Modeling
We provide two deployment modes for VLMs: local deployment and remote API examples  
Run local deployment  
`python src/Image_to_Text.py --mode local`

### Cross-Modal Similarity Verification
Pruning is double-edged - select pruning strategies and thresholds according to task requirements  
Execute verification with  
`python src/cross_modal_verify.py --input ./data --threshold 0.75`

### MMKG Construction for Enhanced Reasoning
We construct MMKG using LightRAG. Before construction, inject image storage paths into generated text  
Add visual locations  
`python src/path_injector.py --input ./text_data --img_dir ./vision_data`  

Build multimodal KG  
`python -m lightrag --input augmented_texts/ --output mmkg/`  

This completes MMKG construction for multimodal reasoning research.

## Citation
```latex
@article{vaLiK2024,
  title={Aligning Vision to Language: Text-Free Multimodal Knowledge Graph Construction for Enhanced LLMs Reasoning},
  author={Your Name, Collaborators},
  journal={GitHub Repository},
  year={2024},
  url={https://github.com/yourname/VaLiK}
}