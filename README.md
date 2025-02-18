# poster-agent

## setup environment

```bash

python -m venv env_poster_agent
env_poster_agent\Scripts\activate
python.exe -m pip install --upgrade pip

pip install numpy==1.26.4
pip install transformers=4.46.2 # need to use this version, or minicpmv will fail

pip install PyQt5
pip install Pillow numpy
pip install llama-index
pip install llama-index-llms-ollama
pip install smolagents[litellm]

pip install "torch>=2.1" "torchvision" "timm>=0.9.2" "transformers>=4.40" "Pillow" "gradio>=4.19" "tqdm" "sentencepiece" "peft" "huggingface-hub>=0.24.0" --extra-index-url https://download.pytorch.org/whl/cpu

pip install "openvino>=2024.4.0" "nncf>=2.12.0"
```