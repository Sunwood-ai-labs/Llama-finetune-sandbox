# 🦙 Unslothで作成したLLaMA 3.2ベースのファインチューニングモデルを使った高速推論ガイド

## 📦 必要なライブラリのインストール

```python
%%capture
!pip install unsloth
# 最新のUnslothナイトリービルドを取得
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**解説**:
Unslothライブラリをインストールします。このライブラリを使用することで、LLaMAモデルのファインチューニングと推論を大幅に高速化できます。ナイトリービルドを使用することで、最新の機能と改善が利用可能です。

## 🔧 ライブラリのインポートと基本設定

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# モデルの基本設定
max_seq_length = 512
dtype = None
load_in_4bit = True
model_id = "MakiAi/Llama-3-2-3B-Instruct-bnb-4bit-10epochs-adapter"  # ファインチューニング済みのモデルパス
```

**解説**:
- 必要なライブラリをインポート
- モデルは4ビット量子化を使用して、メモリ効率を改善
- `model_id`には、Unslothでファインチューニングしたモデルのパスを指定

## 🚀 モデルとトークナイザーの初期化

```python
# モデルとトークナイザーのロード
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    trust_remote_code=True,
)

# LLaMA 3.1のチャットテンプレートを使用
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3.1",  # LLaMA 3.1のテンプレートで問題なし
)

# 高速推論モードを有効化
FastLanguageModel.for_inference(model)  # 通常の2倍の速度
```

**解説**:
1. ファインチューニング済みのモデルをロード
2. LLaMA 3.1のチャットテンプレートを適用（3.2でも互換性あり）
3. Unslothの高速推論モードを有効化

## 💬 データセットを使用した推論の実装

```python
def generate_response(dataset_entry):
    """
    データセットのエントリーに対して応答を生成する関数
    """
    # メッセージの作成
    messages = [
        {"role": "user", "content": dataset_entry["conversations"][0]['content']},
    ]

    # チャットテンプレートの適用
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,  # 生成プロンプトの追加
        return_tensors="pt",
    ).to(model.device)

    # 応答の生成
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=64,  # 生成するトークン数
        use_cache=True,     # キャッシュを使用して高速化
        temperature=1.5,    # より創造的な応答を生成
        min_p=0.1          # 出力の多様性を確保
    )

    return tokenizer.batch_decode(outputs)
```

**解説**:
この関数は：
1. データセットのエントリーからユーザーの入力を抽出
2. LLaMA 3.1形式のチャットテンプレートを適用
3. 以下のパラメータで応答を生成：
   - `max_new_tokens`: 64（短めの応答を生成）
   - `temperature`: 1.5（創造性を高める）
   - `min_p`: 0.1（多様な応答を確保）

## ✅ 実行例

```python
if __name__ == "__main__":
    # テストデータセット
    dataset = [
        {"conversations": [{"content": "火焔猫燐について教えてください。"}]},
        {"conversations": [{"content": "水橋パルスィの本質は何ですか？"}]},
        {"conversations": [{"content": "プログラミング初心者へのアドバイスをお願いします。"}]}
    ]

    # 2番目のデータセットエントリーで試してみる
    response = generate_response(dataset[0])

    print("入力:", dataset[0]["conversations"][0]['content'])
    print("応答:", response)
```

```python
if __name__ == "__main__":
    # テストデータセット
    dataset = [
        {"conversations": [{"content": "火焔猫燐について教えてください。"}]},
        {"conversations": [{"content": "水橋パルスィの本質は何ですか？"}]},
        {"conversations": [{"content": "プログラミング初心者へのアドバイスをお願いします。"}]}
    ]

    # 2番目のデータセットエントリーで試してみる
    response = generate_response(dataset[1])

    print("入力:", dataset[1]["conversations"][0]['content'])
    print("応答:", response)
```

```python
if __name__ == "__main__":
    # テストデータセット
    dataset = [
        {"conversations": [{"content": "火焔猫燐について教えてください。"}]},
        {"conversations": [{"content": "水橋パルスィの本質は何ですか？"}]},
        {"conversations": [{"content": "プログラミング初心者へのアドバイスをお願いします。"}]}
    ]

    # 2番目のデータセットエントリーで試してみる
    response = generate_response(dataset[2])

    print("入力:", dataset[2]["conversations"][0]['content'])
    print("応答:", response)
```

**解説**:
サンプルの実行方法を示しています：
- テスト用のデータセットを定義
- 選択したエントリーで応答を生成
- 入力と生成された応答を表示

このコードを使用することで、UnslothでファインチューニングしたカスタムのデータセットでトレーニングしたLLaMA 3.2モデルを、高速に推論できます。LLaMA 3.1のトークナイザーを使用することで、新しいモデルでも安定した出力が得られます。必要に応じて生成パラメータを調整することで、モデルの応答特性をカスタマイズできます。
