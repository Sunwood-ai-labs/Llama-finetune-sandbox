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
  <img src="docs/Llama-finetune-sandbox.png" width="100%">
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
  ～ Llama Model Fine-tuning Experiment Environment ～
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge&logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Docker-blue?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/CUDA-11.7%2B-green?style=for-the-badge&logo=nvidia" alt="CUDA">
</p>

## 🚀 Project Overview

**Llama-finetune-sandbox** is an experimental environment for learning and verifying fine-tuning of Llama models.  You can try various fine-tuning methods, customize models, and evaluate their performance. It caters to a wide range of users, from beginners to researchers.

## ✨ Key Features

1. **Diverse Fine-tuning Methods**:
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)
   - Full Fine-tuning
   - Parameter-Efficient Fine-tuning (PEFT)

2. **Flexible Model Settings**:
   - Customizable maximum sequence length
   - Various quantization options
   - Multiple attention mechanisms

3. **Experiment Environment Setup**:
   - Performance evaluation tools
   - Memory usage optimization
   - Visualization of experimental results

## 📚 Implementation Examples

This repository includes the following implementation examples:

1. **High-speed fine-tuning using Unsloth**:
   - Implementation of high-speed fine-tuning for Llama-3.2-1B/3B models.
     - → See [`Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md`](sandbox/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md) for details. (Japanese)
     - → [Use this to convert from markdown to notebook format](https://huggingface.co/spaces/MakiAi/JupytextWebUI)
   - [📒Notebook here](https://colab.research.google.com/drive/1AjtWF2vOEwzIoCMmlQfSTYCVgy4Y78Wi?usp=sharing)

2. Other implementation examples will be added periodically.

## 🛠️ Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox.git
cd Llama-finetune-sandbox
```

## 📝 Adding Experiment Examples

1. Add a new implementation to the `examples/` directory.
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

## ⚖️ License

This project is licensed under the MIT License.