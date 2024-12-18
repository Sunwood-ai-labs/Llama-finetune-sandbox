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

**Llama-finetune-sandbox**は、Llamaモデルのファインチューニングを実験的に学習・検証できる環境です。様々なファインチューニング手法を試し、モデルのカスタマイズや性能評価を行うことができます。初学者から研究者まで、幅広いユーザーのニーズに対応します。バージョン0.7.0では、ドキュメントの更新と、Unslothを用いた高速推論実装ガイドの追加を行いました。


## ✨ 主な機能

1. **多様なファインチューニング手法**:
   - LoRA (Low-Rank Adaptation)
   - QLoRA (Quantized LoRA)
   
2. **柔軟なモデル設定**:
   - カスタム可能な最大シーケンス長
   - 多様な量子化オプション
   - 複数のアテンションメカニズム

3. **実験環境の整備**:
   - メモリ使用量の最適化
   - 実験結果の可視化

4. **コンテキストアウェアリフレクティブQA生成システム**:
    - Wikipediaデータから高品質なQ&Aデータセットを生成します。
    - LLMを活用し、文脈を考慮した質問と回答の生成、品質評価、段階的な改善を自動で行います。
    - 事実性、質問の質、回答の完全性を数値化して評価し、段階的に改善を行うリフレクティブなアプローチを採用しています。
    - 環境構築、モデル選択、データ前処理、Q&Aペア生成、品質評価、改善プロセスを網羅したコードと解説を提供しています。
    - `litellm`, `wikipedia`, `transformers`などのライブラリを使用しています。
    - 出力されたQ&AペアはJSON形式で保存され、Hugging Face Hubへのアップロードも容易に行えます。

5. **LLM評価システム**:
    - LLMの回答品質を自動的に評価するシステムです。
    - 質問、模範解答、LLMの回答を4段階で評価し、詳細な評価レポートを生成します。
    - エラーハンドリング、リトライ機能、ログ機能、カスタマイズ可能な評価基準、CSVとHTML形式でのレポート生成機能を備えています。
    - HuggingFace Hubへのアップロード機能も備えています。


## 🔧 使用方法

本リポジトリ内のノートブックを参照してください。

## 📦 インストール手順

`requirements.txt`を参照し、必要なパッケージをインストールしてください。


## 📚 実装例

本リポジトリには以下の実装例が含まれています：

### Unslothを使用した高速ファインチューニング
 - Llama-3.2-1B/3Bモデルの高速ファインチューニング実装  
   - → 詳細は [`Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md`](sandbox/Llama_3_2_1B+3B_Conversational_+_2x_faster_finetuning_JP.md) をご参照ください。
   - → [マークダウン形式からノートブック形式への変換はこちらを使用してください](https://huggingface.co/spaces/MakiAi/JupytextWebUI)
 - [📒ノートブックはこちら](https://colab.research.google.com/drive/1AjtWF2vOEwzIoCMmlQfSTYCVgy4Y78Wi?usp=sharing)

### Unslothを使用した高速推論
 - Llama-3.2モデルの高速推論実装
   - → 詳細は [`Unsloth_inference_llama3-2.md`](sandbox/Unsloth_inference_llama3-2.md) をご参照ください。
   - → Unslothを使用したLlama-3.2モデルの効率的な推論処理の実装
 - [📒ノートブックはこちら](https://colab.research.google.com/drive/1FkAYiX2fbGPTRUopYw39Qt5UE2tWJRpa?usp=sharing)

 - LLM-JPモデルの高速推論実装
   - → 詳細は [`Unsloth_inference_llm_jp.md`](sandbox/Unsloth_inference_llm_jp.md) をご参照ください。
   - → 日本語LLMの高速推論処理の実装とパフォーマンス最適化
 - [📒ノートブックはこちら](https://colab.research.google.com/drive/1lbMKv7NzXQ1ynCg7DGQ6PcCFPK-zlSEG?usp=sharing)

### OllamaとLiteLLMを使用した効率的なモデル運用
 - Google Colabでのセットアップと運用ガイド
 - → 詳細は [`efficient-ollama-colab-setup-with-litellm-guide.md`](sandbox/efficient-ollama-colab-setup-with-litellm-guide.md) をご参照ください。
 - [📒ノートブックはこちら](https://colab.research.google.com/drive/1buTPds1Go1NbZOLlpG94VG22GyK-F4GW?usp=sharing)

### WikipediaデータからのQ&Aデータセット生成（センテンスプールQA方式）
- センテンスプールQA方式による高品質Q&Aデータセット生成
  - → 句点区切りの文をプールして文脈を保持しながらQ&Aペアを生成する新しいデータセット作成手法
  - → チャンクサイズを柔軟に調整可能（デフォルト200文字）で、用途に応じた最適な文脈範囲でQ&Aペアを生成
  - → 詳細は [`wikipedia-qa-dataset-generator.md`](sandbox/wikipedia-qa-dataset-generator.md) をご参照ください。
- [📒ノートブックはこちら](https://colab.research.google.com/drive/1mmK5vxUzjk3lI6OnEPrQqyjSzqsEoXpk?usp=sharing)

### コンテキストアウェアリフレクティブQA生成システム
- リフレクティブな品質改善を行うQ&Aデータセット生成
  - → 生成したQ&Aペアの品質を自動評価し、段階的に改善を行う新方式
  - → 事実性、質問品質、回答の完全性を数値化して評価
  - → 文脈情報を活用した高精度な質問生成と回答の整合性チェック
  - → 詳細は [`context_aware_Reflexive_qa_generator_V2.md`](sandbox/context_aware_Reflexive_qa_generator_V2.md) をご参照ください。
- [📒ノートブックはこちら](https://colab.research.google.com/drive/1OYdgAuXHbl-0LUJgkLl_VqknaAEmAm0S?usp=sharing)

### LLM評価システム (LLMs as a Judge)
- LLMを評価者として活用する高度な品質評価システム
  - → 質問、模範解答、LLMの回答を4段階スケールで自動評価
  - → エラーハンドリングとリトライ機能による堅牢な設計
  - → CSV、HTML形式での詳細な評価レポート生成
  - → 詳細は [`LLMs_as_a_Judge_TOHO_V2.md`](sandbox/LLMs_as_a_Judge_TOHO_V2.md) をご参照ください。
- [📒ノートブックはこちら](https://colab.research.google.com/drive/1Zjw3sOMa2v5RFD8dFfxMZ4NDGFoQOL7s?usp=sharing)


## 🆕 最新情報 (v0.7.0)

- **Unslothを用いた高速推論実装ガイドの追加**: Llama-3.2モデルとLLM-JPモデルの高速推論実装に関する情報、各モデルの使用方法、Colabノートブックへのリンクを追加しました。
- ドキュメントの更新


## 🤝 コントリビューション

- 新しいファインチューニング手法の実装
- バグ修正や機能改善
- ドキュメントの改善
- 使用例の追加

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。