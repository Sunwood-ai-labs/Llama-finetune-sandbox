---
title: Llama-finetune-sandbox
emoji: 🦙
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

<p align="center">
  <img src="https://raw.githubusercontent.com/Sunwood-ai-labs/Llama-finetune-sandbox/refs/heads/main/docs/Llama-finetune-sandbox.png" width="100%">
  <h1 align="center">🌟 Llama-finetune-sandbox 🌟</h1>
</p>

<p align="center">
  <a href="https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox">
    <img alt="GitHub Repo" src="https://img.shields.io/badge/github-Llama--finetune--sandbox-blue?logo=github">
  </a>
  <a href="https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/Sunwood-ai-labs/Llama-finetune-sandbox?color=green">
  </a>
  <a href="https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/Sunwood-ai-labs/Llama-finetune-sandbox?style=social">
  </a>
  <a href="https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/v/release/Sunwood-ai-labs/Llama-finetune-sandbox?include_prereleases&style=flat-square">
  </a>
</p>

<h2 align="center">
  Llama Model Fine-tuning Experiment Environment
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge&logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Docker-blue?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/CUDA-11.7%2B-green?style=for-the-badge&logo=nvidia" alt="CUDA">
</p>

## 🚀 Project Overview

**Llama-finetune-sandbox** provides an experimental environment for learning and verifying Llama model fine-tuning.  You can try various fine-tuning methods, customize models, and evaluate performance.  It caters to a wide range of users, from beginners to researchers. Version 0.2.0 includes updated and improved documentation.  In particular,  guides for efficient model operation using Ollama and LiteLLM have been added, and the README.md and implementation example guides have been made easier to understand.


## ✨ Main Features

1. **Diverse Fine-tuning Methods:**
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)

2. **Flexible Model Configuration:**
   - Customizable maximum sequence length
   - Various quantization options
   - Multiple attention mechanisms

3. **Well-equipped Experiment Environment:**
   - Performance evaluation tools
   - Memory usage optimization
   - Experiment result visualization

## 📚 Implementation Examples

This repository includes the following implementation examples:

### High-Speed Fine-tuning using Unsloth
 - High-speed fine-tuning implementation of Llama-3.2-1B/3B models
   - → See [`Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md`](sandbox/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md) for details.
   - → [Use this to convert from markdown format to notebook format](https://huggingface.co/spaces/MakiAi/JupytextWebUI)
 - [📒Notebook here](https://colab.research.google.com/drive/1AjtWF2vOEwzIoCMmlQfSTYCVgy4Y78Wi?usp=sharing)

### Efficient Model Operation using Ollama and LiteLLM
 - Setup and operation guide on Google Colab
 - → See [`efficient-ollama-colab-setup-with-litellm-guide.md`](sandbox/efficient-ollama-colab-setup-with-litellm-guide.md) for details.
 - [📒Notebook here](https://colab.research.google.com/drive/1buTPds1Go1NbZOLlpG94VG22GyK-F4GW?usp=sharing)


## 🛠️ Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox.git
cd Llama-finetune-sandbox
```

## 📝 Adding Implementation Examples

1. Add new implementations to the `examples/` directory.
2. Add necessary settings and utilities to `utils/`.
3. Update documentation and tests.
4. Create a pull request.

## 🤝 Contributions

- Implementation of new fine-tuning methods
- Bug fixes and feature improvements
- Documentation improvements
- Addition of usage examples

## 📚 References

- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [About Llama models](https://github.com/facebookresearch/llama)
- [Fine-tuning best practices](https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox/wiki)

## 📄 License

This project is licensed under the MIT License.
```