
# Unsloth を使った高速なファインチューニング

このノートブックを実行するには、「ランタイム」から「すべてを実行」を選択してください。**無料の** Tesla T4 Google Colab インスタンスで動作します！

ご自身のコンピュータにUnslothをインストールする場合は、[Githubページの手順](https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions)に従ってください。

このノートブックでは以下を学びます：
- [データの準備](#Data)方法 
- [モデルの学習](#Train)方法
- [推論の実行](#Inference)方法
- [モデルの保存](#Save)方法(例：Llama.cpp用)

**[NEW]** Llama-3.1 8b Instructの2倍高速な推論を無料Colabで試してみましょう。[こちら](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)

このノートブックの主な機能：
1. Maxime Labonneの[FineTome 100K](https://huggingface.co/datasets/mlabonne/FineTome-100k)データセットを使用
2. ShareGPTからHuggingFace形式への変換(`standardize_sharegpt`を使用)
3. アシスタントの応答のみの学習(`train_on_responses_only`を使用)
4. UnslothはPython 3.12およびTorch 2.4、すべてのTRL & Xformersバージョンをサポート

```python
%%capture
!pip install unsloth
# 最新のUnslothナイトリービルドを取得
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

サポートしている機能：
* Llama, Mistral, Phi-3, Gemma, Yi, DeepSeek, Qwen, TinyLlama, Vicuna, Open Hermesなど多数のモデル
* 16bit LoRAまたは4bit QLoRA（どちらも2倍高速）
* `max_seq_length`は任意の値に設定可能（[kaiokendev](https://kaiokendev.github.io/til)のRoPE自動スケーリングを使用）
* [**NEW**] Gemma-2 9b / 27bが**2倍高速**に！[Gemma-2 9bのノートブック](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)をご覧ください
* [**NEW**] Ollamaへの自動エクスポート付きファインチューニングは[Ollamaノートブック](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)をお試しください

```python
from unsloth import FastLanguageModel
import torch

# モデルの基本設定
max_seq_length = 2048  # 任意の長さを指定可能（RoPEスケーリングを内部で自動対応）
dtype = None  # 自動検出。Tesla T4, V100はFloat16、Ampere以降はBfloat16
load_in_4bit = True  # メモリ使用量を削減する4bit量子化を使用。必要に応じてFalseに設定可能

# 4bit事前量子化済みの対応モデル一覧（4倍速いダウンロードとメモリ節約）
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2倍速
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 405bモデルの4bit版!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2倍速!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2倍速!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2倍速!
    
    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2モデル群
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
] # より多くのモデルは https://huggingface.co/unsloth で確認できます

# モデルとトークナイザーの読み込み
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",  # または "unsloth/Llama-3.2-1B-Instruct" を選択
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...",  # meta-llama/Llama-2-7b-hf などのゲート制限付きモデルを使用する場合はトークンが必要
)
```

ここで、LoRAアダプターを追加します。これにより、全パラメータの1〜10%のみを更新すれば良くなります！

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # 0より大きい任意の数を選択可能（推奨値: 8, 16, 32, 64, 128）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,  # どの値もサポートしますが、0が最適化されています
    bias = "none",     # どの値もサポートしますが、"none"が最適化されています
    use_gradient_checkpointing = "unsloth",  # 長いコンテキストには True か "unsloth" を使用（"unsloth"は30%メモリ節約）
    random_state = 3407,
    use_rslora = False,   # Rank Stabilized LoRAをサポート
    loftq_config = None,  # LoftQもサポート
)
```

## Data

### データの準備
会話形式のファインチューニングには`Llama-3.1`形式を使用します。今回は[Maxime LabonneのFineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k)データセットをShareGPT形式で使用します。ただし、`("from", "value")`形式ではなく、HuggingFaceの標準的なマルチターン形式`("role", "content")`に変換します。Llama-3は会話を以下のように表示します：

```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hey there! How are you?<|eot_id|><|start_header_id|>user<|end_header_id|>

I'm great thanks!<|eot_id|>
```

適切なチャットテンプレートを取得するために`get_chat_template`関数を使用します。`zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, phi3, llama3`など多数のテンプレートをサポートしています。

```python
from unsloth.chat_templates import get_chat_template

# チャットテンプレートの設定
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# 入力データの整形関数
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# データセットの読み込み
from datasets import load_dataset
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")
```

`standardize_sharegpt`を使用して、ShareGPT形式のデータセットをHuggingFaceの一般的な形式に変換します。
これにより、以下のような形式の変換が行われます：

```python
# 変換前 (ShareGPT形式)：
{"from": "system", "value": "You are an assistant"}
{"from": "human", "value": "What is 2+2?"}
{"from": "gpt", "value": "It's 4."}

# 変換後 (HuggingFace形式)：
{"role": "system", "content": "You are an assistant"}
{"role": "user", "content": "What is 2+2?"}
{"role": "assistant", "content": "It's 4."}
```

```python
from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)
```

会話の構造を確認するために、5番目の会話を見てみましょう：

```python
dataset[5]["conversations"]
```

そして、チャットテンプレートがこれらの会話をどのように変換したかを確認します：

**注意:** Llama 3.1 Instructのデフォルトチャットテンプレートは、`"Cutting Knowledge Date: December 2023\nToday Date: 26 July 2024"`を自動的に追加します。これは正常な動作です。

```python
dataset[5]["text"]
```

## Train

### モデルの学習
HuggingFace TRLの`SFTTrainer`を使用します。詳細は[TRL SFTドキュメント](https://huggingface.co/docs/trl/sft_trainer)を参照してください。

ここでは処理速度を優先して60ステップで実行しますが、完全な学習を行う場合は`num_train_epochs=1`を設定し、`max_steps=None`とすることができます。また、TRLの`DPOTrainer`もサポートしています！

```python
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,  # 短いシーケンスの場合、トレーニングを5倍高速化できます
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1,  # 完全な学習実行の場合はこちらを設定
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",  # WandBなどを使用する場合はここで設定
    ),
)
```

Unslothの`train_on_completions`メソッドを使用して、ユーザーの入力を無視し、アシスタントの出力のみを学習対象とします。

```python
from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
```

マスキングが正しく適用されているか確認してみましょう：

```python
# 入力テキストの確認
tokenizer.decode(trainer.train_dataset[5]["input_ids"])
```

```python
# ラベルの確認
space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])
```

システムプロンプトと指示部分が正しくマスクされていることが確認できます！

```python
#@title 現在のメモリ使用状況を表示
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. 最大メモリ = {max_memory} GB.")
print(f"{start_gpu_memory} GBのメモリが予約済み.")
```

```python
# モデルの学習を実行
trainer_stats = trainer.train()
```

```python
#@title 最終的なメモリと時間の統計を表示
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"学習に使用した時間: {trainer_stats.metrics['train_runtime']} 秒")
print(f"学習に使用した時間: {round(trainer_stats.metrics['train_runtime']/60, 2)} 分")
print(f"最大予約メモリ = {used_memory} GB")
print(f"学習用の最大予約メモリ = {used_memory_for_lora} GB")
print(f"最大メモリに対する使用率 = {used_percentage} %")
print(f"学習用の最大メモリ使用率 = {lora_percentage} %")
```

## Inference

### 推論の実行
モデルを実行してみましょう！指示と入力は変更可能です。出力は空のままにしてください。

**[NEW]** Llama-3.1 8b Instructの2倍高速な推論を無料Colabで試せます。[こちら](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)

`min_p = 0.1`と`temperature = 1.5`を使用します。この設定に関する詳細は[このツイート](https://x.com/menhguin/status/1826132708508213629)をご覧ください。

```python
from unsloth.chat_templates import get_chat_template

# チャットテンプレートの再設定
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model)  # ネイティブの2倍速い推論を有効化

# テスト用の会話を設定
messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,  # 生成には必須
    return_tensors = "pt",
).to("cuda")

# モデルによる生成
outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
tokenizer.batch_decode(outputs)
```


`TextStreamer`を使用することで、生成を待つ間トークンごとに連続的に出力を確認できます。

```python
FastLanguageModel.for_inference(model)  # ネイティブの2倍速い推論を有効化

messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,  # 生成には必須
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
```

## Save

### 学習済みモデルの保存とロード
LoRAアダプターとして最終モデルを保存するには、オンライン保存用のHuggingFaceの`push_to_hub`またはローカル保存用の`save_pretrained`を使用します。

**注意:** これはLoRAアダプターのみを保存し、完全なモデルは保存しません。16bitまたはGGUFとして保存する場合は、以下の説明をご覧ください。

```python
# ローカル保存
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
# オンライン保存
# model.push_to_hub("your_name/lora_model", token = "...")
# tokenizer.push_to_hub("your_name/lora_model", token = "...")
```

保存したLoRAアダプターを推論用にロードする場合は、`False`を`True`に変更してください：

```python
if False:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model",  # 学習に使用したモデル
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )
    FastLanguageModel.for_inference(model)  # ネイティブの2倍速い推論を有効化

# テスト用メッセージ
messages = [
    {"role": "user", "content": "Describe a tall tower in the capital of France."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,  # 生成には必須
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
```

また、HuggingFaceの`AutoModelForPeftCausalLM`を使用することもできます。ただし、これは`unsloth`がインストールされていない場合のみ使用してください。`4bit`モデルのダウンロードがサポートされておらず、Unslothの推論が2倍速いため、非常に遅くなる可能性があります。

### VLLMのためのfloat16形式での保存

モデルを直接`float16`形式で保存することもサポートしています。float16の場合は`merged_16bit`を、int4の場合は`merged_4bit`を選択します。フォールバックオプションとして`lora`アダプターも利用可能です。Hugging Face アカウントにアップロードするには`push_to_hub_merged`を使用します。パーソナルトークンは https://huggingface.co/settings/tokens で取得できます。

```python
# 16bit形式でマージ
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit", token = "")

# 4bit形式でマージ
if False: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_4bit", token = "")

# LoRAアダプターのみ
if False: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
if False: model.push_to_hub_merged("hf/model", tokenizer, save_method = "lora", token = "")
```

### GGUF / llama.cppへの変換

`GGUF` / `llama.cpp`形式での保存も標準でサポートしています！`llama.cpp`をクローンし、デフォルトで`q8_0`形式で保存します。`q4_k_m`などの他の形式もサポートしています。ローカル保存には`save_pretrained_gguf`を、HFへのアップロードには`push_to_hub_gguf`を使用します。

サポートしている量子化メソッド（完全なリストは[Wikiページ](https://github.com/unslothai/unsloth/wiki#gguf-quantization-options)を参照）：
* `q8_0` - 高速な変換。リソース使用量は多いが、一般的に許容範囲
* `q4_k_m` - 推奨。attention.wvとfeed_forward.w2テンソルの半分にQ6_Kを使用し、残りにQ4_Kを使用
* `q5_k_m` - 推奨。attention.wvとfeed_forward.w2テンソルの半分にQ6_Kを使用し、残りにQ5_Kを使用

[**NEW**] Ollamaへの自動エクスポート付きファインチューニングは[Ollamaノートブック](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)をお試しください。

```python
# 8bit Q8_0形式で保存
if False: model.save_pretrained_gguf("model", tokenizer,)
# トークンは https://huggingface.co/settings/tokens で取得してください！
# hfはあなたのユーザー名に変更してください！
if False: model.push_to_hub_gguf("hf/model", tokenizer, token = "")

# 16bit GGUF形式で保存
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "f16", token = "")

# q4_k_m GGUF形式で保存
if False: model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf("hf/model", tokenizer, quantization_method = "q4_k_m", token = "")

# 複数のGGUF形式で保存 - 複数形式が必要な場合はこちらが高速
if False:
    model.push_to_hub_gguf(
        "hf/model",  # hfをあなたのユーザー名に変更してください！
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m",],
        token = "",  # トークンは https://huggingface.co/settings/tokens で取得
    )
```

変換したモデル（`model-unsloth.gguf`または`model-unsloth-Q4_K_M.gguf`）は、`llama.cpp`や`GPT4All`などのUIベースのシステムで使用できます。GPT4Allは[公式サイト](https://gpt4all.io/index.html)からインストール可能です。

**[NEW]** Llama-3.1 8b Instructの2倍高速な推論を無料Colabで試せます。[こちら](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing)から

以上で終了です！Unslothについて質問がある場合は、[Discord](https://discord.gg/u54VK8m8tk)チャンネルをご利用ください。バグの報告や最新のLLM情報の入手、サポートの要請、プロジェクトへの参加などはDiscordで受け付けています。

その他の参考リンク：
1. Zephyr DPO 2倍速 [無料Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)
2. Llama 7b 2倍速 [無料Colab](https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing)
3. TinyLlama 4倍速 Alpaca 52K完全版（1時間） [無料Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)
4. CodeLlama 34b 2倍速 [ColabのA100版](https://colab.research.google.com/drive/1y7A0AxE3y8gdj4AVkl2aZX47Xu3P1wJT?usp=sharing)
5. Mistral 7b [Kaggle版](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook)
6. 🤗 HuggingFaceとの[ブログ記事](https://huggingface.co/blog/unsloth-trl)、TRL[ドキュメント](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth)にも掲載
7. ShareGPTデータセット用の`ChatML`、[会話ノートブック](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing)
8. 小説執筆などのテキスト生成[ノートブック](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing)
9. [**NEW**] Phi-3 Medium / Mini が**2倍速**に！[Phi-3 Mediumノートブック](https://colab.research.google.com/drive/1hhdhBa1j_hsymiW9m-WzxQtgqTH_NHqi?usp=sharing)
10. [**NEW**] Gemma-2 9b / 27b が**2倍速**に！[Gemma-2 9bノートブック](https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing)
11. [**NEW**] Ollamaへの自動エクスポート付きファインチューニング[Ollamaノートブック](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)
12. [**NEW**] Mistral NeMo 12Bが2倍速に！12GB未満のVRAMで動作！[Mistral NeMoノートブック](https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing)
