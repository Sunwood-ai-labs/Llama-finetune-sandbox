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
  Llama model fine-tuning experiment environment
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge&logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Docker-blue?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/CUDA-11.7%2B-green?style=for-the-badge&logo=nvidia" alt="CUDA">
</p>

## 🚀 Project Overview

**Llama-finetune-sandbox** provides an experimental environment for learning and verifying Llama model fine-tuning.  You can try various fine-tuning methods, customize models, and evaluate performance.  It caters to a wide range of users, from beginners to researchers. Version 0.6.0 includes updated documentation and the implementation of an LLM evaluation system. This system automatically assesses the quality of LLM responses and generates detailed evaluation reports.


## ✨ Main Features

1. **Various Fine-tuning Methods:**
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)

2. **Flexible Model Settings:**
   - Customizable maximum sequence length
   - Various quantization options
   - Multiple attention mechanisms

3. **Experiment Environment Setup:**
   - Optimized memory usage
   - Visualization of experimental results

4. **Context-Aware Reflective QA Generation System:**
    - Generates high-quality Q&A datasets from Wikipedia data.
    - Uses LLMs to generate context-aware questions and answers, automatically evaluate quality, and iteratively improve them.
    - Employs a reflective approach, quantifying factuality, question quality, and answer completeness to enable iterative improvements.
    - Provides comprehensive code and explanations covering environment setup, model selection, data preprocessing, Q&A pair generation, quality evaluation, and the improvement process.
    - Uses libraries such as `litellm`, `wikipedia`, and `transformers`.
    - Generated Q&A pairs are saved in JSON format and can be easily uploaded to the Hugging Face Hub.

5. **LLM Evaluation System:**
    - Automatically evaluates the quality of LLM responses.
    - Evaluates questions, model answers, and LLM responses on a 4-level scale, generating detailed evaluation reports.
    - Features error handling, retry functionality, logging, customizable evaluation criteria, and report generation in CSV and HTML formats.
    - Also includes functionality for uploading to the HuggingFace Hub.


## 🔧 Usage

Please refer to the notebooks in this repository.


## 📦 Installation Instructions

Refer to `requirements.txt` and install the necessary packages.


## 📚 Examples

This repository includes the following examples:

### Fast Fine-tuning using Unsloth
 - Fast fine-tuning implementation for Llama-3.2-1B/3B models
   - → See [`Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md`](sandbox/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md) for details.
   - → [Use this to convert from markdown to notebook format](https://huggingface.co/spaces/MakiAi/JupytextWebUI)
 - [📒Notebook here](https://colab.research.google.com/drive/1AjtWF2vOEwzIoCMmlQfSTYCVgy4Y78Wi?usp=sharing)

### Fast Inference using Unsloth
 - Fast inference implementation for Llama-3.2 models
   - → See [`Unsloth_inference_llama3-2.md`](sandbox/Unsloth_inference_llama3-2.md) for details.
   - → Implementation of efficient inference processing for Llama-3.2 models using Unsloth
 - [📒Notebook here](https://colab.research.google.com/drive/1FkAYiX2fbGPTRUopYw39Qt5UE2tWJRpa?usp=sharing)

 - Fast inference implementation for LLM-JP models
   - → See [`Unsloth_inference_llm_jp.md`](sandbox/Unsloth_inference_llm_jp.md) for details.
   - → Implementation and performance optimization of fast inference processing for Japanese LLMs
 - [📒Notebook here](https://colab.research.google.com/drive/1lbMKv7NzXQ1ynCg7DGQ6PcCFPK-zlSEG?usp=sharing)

### Efficient Model Operation using Ollama and LiteLLM
 - Setup and operation guide on Google Colab
 - → See [`efficient-ollama-colab-setup-with-litellm-guide.md`](sandbox/efficient-ollama-colab-setup-with-litellm-guide.md) for details.
 - [📒Notebook here](https://colab.research.google.com/drive/1buTPds1Go1NbZOLlpG94VG22GyK-F4GW?usp=sharing)

### Q&A Dataset Generation from Wikipedia Data (Sentence Pool QA Method)
- High-quality Q&A dataset generation using the Sentence Pool QA method
  - → A new dataset creation method that generates Q&A pairs while preserving context by pooling sentences separated by periods.
  - → Chunk size can be flexibly adjusted (default 200 characters) to generate Q&A pairs with optimal context ranges for various applications.
  - → See [`wikipedia-qa-dataset-generator.md`](sandbox/wikipedia-qa-dataset-generator.md) for details.
- [📒Notebook here](https://colab.research.google.com/drive/1mmK5vxUzjk3lI6OnEPrQqyjSzqsEoXpk?usp=sharing)

### Context-Aware Reflective QA Generation System
- Q&A dataset generation with reflective quality improvement
  - → A new method that automatically evaluates the quality of generated Q&A pairs and iteratively improves them.
  - → Quantifies factuality, question quality, and answer completeness for evaluation.
  - → Uses contextual information for high-accuracy question generation and answer consistency checks.
  - → See [`context_aware_Reflexive_qa_generator_V2.md`](sandbox/context_aware_Reflexive_qa_generator_V2.md) for details.
- [📒Notebook here](https://colab.research.google.com/drive/1OYdgAuXHbl-0LUJgkLl_VqknaAEmAm0S?usp=sharing)

### LLM Evaluation System (LLMs as a Judge)
- Advanced quality evaluation system utilizing LLMs as evaluators
  - → Automatically evaluates questions, model answers, and LLM responses on a 4-level scale.
  - → Robust design with error handling and retry functionality.
  - → Generates detailed evaluation reports in CSV and HTML formats.
  - → See [`LLMs_as_a_Judge_TOHO_V2.md`](sandbox/LLMs_as_a_Judge_TOHO_V2.md) for details.
- [📒Notebook here](https://colab.research.google.com/drive/1Zjw3sOMa2v5RFD8dFfxMZ4NDGFoQOL7s?usp=sharing)


## 🆕 Latest Information (v0.6.0)

- **Implementation of the LLM Evaluation System:** Added a system that automatically evaluates the quality of LLM responses.  Questions, model answers, and LLM responses are compared and evaluated on a 4-level scale.  Features error handling, retry functionality, logging, customizable evaluation criteria, and report generation in CSV and HTML formats.
- Added information about the LLM evaluation system to README.md


## 🤝 Contributions

- Implementation of new fine-tuning methods
- Bug fixes and feature improvements
- Documentation improvements
- Addition of usage examples

## 📄 License

This project is licensed under the MIT License.