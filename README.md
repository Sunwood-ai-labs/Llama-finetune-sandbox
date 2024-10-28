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
  ～ Llamaモデルのファインチューニング実験環境 ～
</h2>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?style=for-the-badge&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/HuggingFace-yellow?style=for-the-badge&logo=huggingface" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Docker-blue?style=for-the-badge&logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/CUDA-11.7%2B-green?style=for-the-badge&logo=nvidia" alt="CUDA">
</p>

## 🚀 プロジェクト概要

**Llama-finetune-sandbox**は、Llamaモデルのファインチューニングを実験的に学習・検証できる環境です。様々なファインチューニング手法を試し、モデルのカスタマイズや性能評価を行うことができます。初学者から研究者まで、幅広いユーザーのニーズに対応します。バージョン0.1.0では、リポジトリ名が変更され、READMEが大幅に更新されました。さらに、Llamaモデルのファインチューニングチュートリアルが追加されました。

## ✨ 主な機能

1. **多様なファインチューニング手法**:
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)
   - ⚠️~Full Fine-tuning~
   - ⚠️~Parameter-Efficient Fine-tuning (PEFT)~
   
2. **柔軟なモデル設定**:
   - カスタム可能な最大シーケンス長
   - 多様な量子化オプション
   - 複数のアテンションメカニズム

3. **実験環境の整備**:
   - 性能評価ツール
   - メモリ使用量の最適化
   - 実験結果の可視化

## 🔧 使用方法

本リポジトリには、Unslothライブラリを使用した高速ファインチューニングのチュートリアル(`sandbox/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md`)が含まれています。このチュートリアルでは、ステップバイステップでファインチューニングの方法を説明し、コード例も豊富に含んでいます。日本語で記述されたチュートリアルです。[マークダウン形式からノートブック形式への変換はこちらを使用してください](https://huggingface.co/spaces/MakiAi/JupytextWebUI)  また、[Google Colabノートブック](https://colab.research.google.com/drive/1AjtWF2vOEwzIoCMmlQfSTYCVgy4Y78Wi?usp=sharing)もご利用いただけます。


## 📦 インストール手順

情報がありません。


## 🆕 最新情報

- 🎉 Llamaモデルファインチューニングチュートリアルの追加。


## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。
