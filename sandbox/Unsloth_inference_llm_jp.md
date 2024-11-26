# 🤖 UnslothによるLLM-JPモデルの高速推論実装ガイド(Google Colab📒ノートブック付)

## 📦 必要なライブラリのインストール

```python
%%capture
!pip install unsloth
# 最新のUnslothナイトリービルドを取得
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**解説**:
このセルではUnslothライブラリをインストールしています。Unslothは大規模言語モデル（LLM）の推論を高速化するためのライブラリです。`%%capture`を使用することで、インストール時の出力を非表示にしています。

## 🔧 必要なライブラリのインポートと基本設定

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel
import torch

# モデルの基本設定
max_seq_length = 512
dtype = None
load_in_4bit = True
model_id = "llm-jp/llm-jp-3-13b"  # または自分でファインチューニングしたモデルID
```

**解説**:
- `transformers`: Hugging Faceの変換器ライブラリ
- `unsloth`: 高速化ライブラリ
- `torch`: PyTorchフレームワーク
- モデルの設定では：
  - 最大シーケンス長: 512トークン
  - 4ビット量子化を有効化
  - LLM-JP 13Bモデルを使用

## 🚀 モデルとトークナイザーの初期化

```python
# モデルとトークナイザーのロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
)

# 推論モードに設定
FastLanguageModel.for_inference(model)
```

**解説**:
このセルでは：
1. モデルとトークナイザーを同時にロード
2. 4ビット量子化を適用し、メモリ使用量を削減
3. モデルを推論モードに設定して最適化

## 💬 応答生成関数の実装

```python
def generate_response(input_text):
    """
    入力テキストに対して応答を生成する関数
    """
    # プロンプトの作成
    prompt = f"""### 指示\n{input_text}\n### 回答\n"""

    # 入力のトークナイズ
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    # 応答の生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        use_cache=True,
        do_sample=False,
        repetition_penalty=1.2
    )

    # デコードして回答部分を抽出
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n### 回答')[-1]

    return prediction
```

**解説**:
この関数は以下の処理を行います：
1. 入力テキストを指示形式のプロンプトに変換
2. トークナイズしてモデルに入力可能な形式に変換
3. 以下のパラメータで応答を生成：
   - `max_new_tokens`: 最大512トークンまで生成
   - `use_cache`: キャッシュを使用して高速化
   - `do_sample`: 決定的な出力を生成
   - `repetition_penalty`: 繰り返しを抑制（1.2）
4. 生成された出力から回答部分のみを抽出

## ✅ 使用例

```python
if __name__ == "__main__":
    # 入力例
    sample_input = "今日の天気について教えてください。"

    # 応答の生成
    response = generate_response(sample_input)

    print("入力:", sample_input)
    print("応答:", response)
```

**解説**:
このセルは実際の使用例を示しています：
- サンプル入力を設定
- `generate_response`関数を呼び出して応答を生成
- 入力と応答を表示

このコードを実行することで、LLM-JPモデルを使用して日本語の質問に対する応答を生成できます。Unslothによる最適化により、標準的な実装と比較して高速な推論が可能です。


