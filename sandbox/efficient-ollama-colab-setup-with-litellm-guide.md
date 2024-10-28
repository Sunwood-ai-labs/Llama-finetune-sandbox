# LiteLLMを活用してOllamaをGoogle Colabで効率的に運用する方法

## はじめに

ローカルLLMの運用において、OllamaとLiteLLMの組み合わせは非常に強力なソリューションとなっています。本記事では、Google Colab環境でこれらのツールを効率的に統合する方法を解説します。

## Ollamaとは

Ollamaは、ローカル環境でLLM（大規模言語モデル）を簡単に実行できるオープンソースのツールです。主な特徴として：

- 簡単なコマンドラインインターフェース
- 効率的なモデル管理
- 軽量な実行環境
- APIサーバーとしての機能

## LiteLLMを使う利点

LiteLLMを導入することで得られる主なメリット：

1. **統一されたインターフェース**
   - OpenAI
   - Anthropic
   - Ollama
   - その他の主要なLLMプロバイダーに同じコードで接続可能

2. **容易なプロバイダー切り替え**
   - モデルの指定を変更するだけで異なるプロバイダーに切り替え可能
   - 開発環境とプロダクション環境での柔軟な切り替え

3. **標準化されたエラーハンドリング**
   - 各プロバイダー固有のエラーを統一的に処理

## 実装手順

### 環境のセットアップ

```python
# Ollamaのインストール
!curl https://ollama.ai/install.sh | sh

# CUDAドライバーのインストール
!echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
!sudo apt-get update && sudo apt-get install -y cuda-drivers
```

### サーバーの起動とモデルのダウンロード

```python
# Ollamaサーバーの起動
!nohup ollama serve &

# モデルのダウンロード
!ollama pull llama3:8b-instruct-fp16
```

### LiteLLMを使用したモデル実行

```python
from litellm import completion

response = completion(
    model="ollama/llama3:8b-instruct-fp16", 
    messages=[{ "content": "respond in 20 words. who are you?","role": "user"}], 
    api_base="http://localhost:11434"
)
print(response)
```

## プロバイダーの切り替え例

LiteLLMを使用することで、以下のように簡単に異なるプロバイダーに切り替えることができます：

```python
# OpenAIの場合
response = completion(
    model="gpt-3.5-turbo", 
    messages=[{"content": "Hello!", "role": "user"}]
)

# Anthropicの場合
response = completion(
    model="claude-3-opus-20240229", 
    messages=[{"content": "Hello!", "role": "user"}]
)

# Ollamaの場合（ローカル実行）
response = completion(
    model="ollama/llama3:8b-instruct-fp16", 
    messages=[{"content": "Hello!", "role": "user"}],
    api_base="http://localhost:11434"
)
```

## 注意点とベストプラクティス

1. **リソース管理**
   - Google Colabの無料枠でも実行可能
   - GPUメモリの使用状況に注意

2. **セッション管理**
   - Colabのセッション切断時は再セットアップが必要
   - 長時間の実行にはPro版の使用を推奨

## まとめ

OllamaとLiteLLMの組み合わせは、ローカルLLMの運用を大幅に簡素化します。特に：

- 統一されたインターフェースによる開発効率の向上
- 異なるプロバイダー間での容易な切り替え
- Google Colab環境での簡単な実行

これらの利点により、プロトタイピングから本番環境まで、柔軟なLLMの活用が可能となります。

## ノートブック

https://colab.research.google.com/drive/1buTPds1Go1NbZOLlpG94VG22GyK-F4GW?usp=sharing

## リポジトリ

https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox

## 参考サイト

https://note.com/masayuki_abe/n/n9640e08492ac

