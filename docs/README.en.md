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
  Llama Model Fine-tuning Experimentation Environment
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge&logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Docker-blue?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/CUDA-11.7%2B-green?style=for-the-badge&logo=nvidia" alt="CUDA">
</p>

## 🚀 Project Overview

**Llama-finetune-sandbox** provides an experimental environment for learning and verifying the fine-tuning of Llama models.  You can try various fine-tuning methods, customize models, and evaluate performance.  It caters to a wide range of users, from beginners to researchers. Version 0.5.0 includes updated documentation and the addition of a context-aware reflexive QA generation system. This system generates high-quality Q&A datasets from Wikipedia data, leveraging LLMs to iteratively improve the quality of questions and answers, resulting in a more accurate dataset.


## ✨ Key Features

1. **Diverse Fine-tuning Methods:**
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)

2. **Flexible Model Configuration:**
   - Customizable maximum sequence length
   - Various quantization options
   - Multiple attention mechanisms

3. **Well-equipped Experimentation Environment:**
   - Optimized memory usage
   - Visualization of experimental results

4. **Context-Aware Reflexive QA Generation System:**
    - Generates high-quality Q&A datasets from Wikipedia data.
    - Uses LLMs to automatically generate context-aware questions and answers, evaluate quality, and iteratively improve them.
    - Employs a reflexive approach, quantifying factuality, question quality, and answer completeness for iterative improvement.
    - Provides comprehensive code and explanations covering environment setup, model selection, data preprocessing, Q&A pair generation, quality evaluation, and the improvement process.
    - Utilizes libraries such as `litellm`, `wikipedia`, and `transformers`.
    - Generated Q&A pairs are saved in JSON format and can be easily uploaded to the Hugging Face Hub.


## 📚 Examples

This repository includes the following examples:

### Fast Fine-tuning using Unsloth
 - Implementation of fast fine-tuning for Llama-3.2-1B/3B models.
   - → See [`Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md`](sandbox/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md) for details.
   - → [Use this to convert from markdown to notebook format](https://huggingface.co/spaces/MakiAi/JupytextWebUI)
 - [📒Notebook here](https://colab.research.google.com/drive/1AjtWF2vOEwzIoCMmlQfSTYCVgy4Y78Wi?usp=sharing)

### Efficient Model Deployment using Ollama and LiteLLM
 - Setup and deployment guide on Google Colab.
 - → See [`efficient-ollama-colab-setup-with-litellm-guide.md`](sandbox/efficient-ollama-colab-setup-with-litellm-guide.md) for details.
 - [📒Notebook here](https://colab.research.google.com/drive/1buTPds1Go1NbZOLlpG94VG22GyK-F4GW?usp=sharing)

### Q&A Dataset Generation from Wikipedia Data (Sentence Pool QA Method)
- High-quality Q&A dataset generation using the sentence pool QA method.
  - → A new dataset creation method that generates Q&A pairs while preserving context by pooling sentence segments delimited by punctuation.
  - → Chunk size is flexibly adjustable (default 200 characters) to generate Q&A pairs with an optimal context range depending on the application.
  - → See [`wikipedia-qa-dataset-generator.md`](sandbox/wikipedia-qa-dataset-generator.md) for details.
- [📒Notebook here](https://colab.research.google.com/drive/1mmK5vxUzjk3lI6OnEPrQqyjSzqsEoXpk?usp=sharing)

### Context-Aware Reflexive QA Generation System
- Q&A dataset generation with reflexive quality improvement.
  - → Automatically evaluates the quality of generated Q&A pairs and iteratively improves them.
  - → Quantifies factuality, question quality, and answer completeness for evaluation.
  - → Generates high-precision questions and performs consistency checks on answers using contextual information.
  - → See [`context_aware_Reflexive_qa_generator_V2.md`](sandbox/context_aware_Reflexive_qa_generator_V2.md) for details.
- [📒Notebook here](https://colab.research.google.com/drive/1OYdgAuXHbl-0LUJgkLl_VqknaAEmAm0S?usp=sharing)

### LLM Evaluation System (LLMs as a Judge)
- Advanced quality evaluation system using LLMs as evaluators.
  - → Automatically evaluates questions, model answers, and LLM responses on a four-level scale.
  - → Robust design with error handling and retry functions.
  - → Generates detailed evaluation reports in CSV and HTML formats.
  - → See [`LLMs_as_a_Judge_TOHO_V2.md`](sandbox/LLMs_as_a_Judge_TOHO_V2.md) for details.
- [📒Notebook here](https://colab.research.google.com/drive/1Zjw3sOMa2v5RFD8dFfxMZ4NDGFoQOL7s?usp=sharing)


## 🛠️ Setup

1. Clone the repository:
```bash
git clone https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox.git
cd Llama-finetune-sandbox
```

## 📝 Adding Examples

1. Add new implementations to the `sandbox/` directory.
2. Add necessary settings and utilities to `utils/` (This section was removed as `utils/` directory appears not to exist).
3. Update documentation and tests (This section was removed as there's no mention of existing tests).
4. Create a pull request.

## 🤝 Contributions

- Implementation of new fine-tuning methods
- Bug fixes and feature improvements
- Documentation improvements
- Addition of usage examples

## 📚 References

- [HuggingFace PEFT documentation](https://huggingface.co/docs/peft)
- [About Llama models](https://github.com/facebookresearch/llama)
- [Fine-tuning best practices](https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox/wiki) (This section was removed as the wiki page appears not to exist).

## 📄 License

This project is licensed under the MIT License.

## v0.5.0 Updates

**🆕 What's New:**

- Implementation of the context-aware reflexive QA generation system.
- Addition of relevant information to README.md.