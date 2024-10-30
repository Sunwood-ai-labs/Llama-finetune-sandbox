# WikipediaデータからLlama 3.1用Q&Aデータセット生成ツールの解説（ Google Colabノートブック付）

## はじめに

このツールは、Wikipediaの記事からLlama 3.1のファインチューニング用Q&Aデータセットを自動生成するためのものです。生成されたデータセットは、Llama 3.1の会話形式に準拠しており、高品質な学習データとして活用できます。

## システム構成

このツールは以下の4つの主要クラスで構成されています：

1. WikiTextProcessor - Wikipedia記事の取得と前処理
2. QAGenerator - Q&Aペアの生成とLlama形式への変換
3. QualityChecker - 生成されたQ&Aペアの品質管理
4. DatasetCreator - 全体のプロセス管理

## セットアップと環境構築

### 必要なライブラリのインストール

```python
!pip install litellm tqdm loguru wikipedia
```

### モデルの設定

```python
MODEL_NAME = "ollama/llama3.1:8b-instruct-fp16"
```

## 主要コンポーネントの解説

### WikiTextProcessor クラス

このクラスはWikipediaのテキストデータを取得し、適切な形式に加工します。

```python
class WikiTextProcessor:
    @staticmethod
    def get_wiki_text(topic: str, lang: str = "ja") -> str:
        """指定されたトピックのWikipedia記事を取得"""
        wikipedia.set_lang(lang)
        try:
            page = wikipedia.page(topic)
            return page.content
        except Exception as e:
            logger.error(f"Error fetching {topic}: {e}")
            return ""
            
    @staticmethod
    def clean_text(text: str) -> str:
        """テキストのクリーニング"""
        import re
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()
        
    @staticmethod
    def split_into_chunks(text: str, max_chunk_size: int = 200) -> List[str]:
        """テキストを意味のある単位で分割"""
        sentences = text.split('。')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence) + 1
            if current_size + sentence_size > max_chunk_size and current_chunk:
                chunks.append('。'.join(current_chunk) + '。')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunks.append('。'.join(current_chunk) + '。')
            
        return chunks
```

このクラスの主な機能：
- Wikipedia記事の取得
- テキストのクリーニング（参照記号の除去など）
- 長い記事の適切なサイズへの分割

### QAGenerator クラス

Q&Aペアの生成とLlama形式への変換を担当します。

```python
class QAGenerator:
    @staticmethod
    def generate_qa_pairs_with_retry(
        chunk: str,
        num_pairs: int = 5,
        max_retries: int = 3,
        base_delay: int = 2
    ) -> List[Dict[str, str]]:
        prompt = f"""
        以下のテキストから質問と回答のペアを{num_pairs}つ生成してください。
        必ず以下の条件をすべて満たすJSONを出力してください：

        1. 厳密なJSON形式で出力（最後の要素にもカンマをつけない）
        2. すべての質問は日本語で記述
        3. すべての質問は「？」で終わる
        4. 回答は500文字以下
        5. 回答は必ずテキストの中に記載されている内容にして

        テキスト:
        {chunk}

        出力形式:
        {{
            "qa_pairs": [
                {{"question": "質問1？", "answer": "回答1"}},
                {{"question": "質問2？", "answer": "回答2"}},
                {{"question": "質問3？", "answer": "回答3"}}
            ]
        }}
        """
        # ... リトライロジックの実装 ...
```

主な機能：
- Q&Aペアの生成（リトライ機能付き）
- JSONからの応答抽出
- エラー処理と再試行メカニズム

### QualityChecker クラス

生成されたQ&Aペアの品質管理を行います。

```python
class QualityChecker:
    @staticmethod
    def validate_qa_pair(qa_pair: Dict[str, str]) -> bool:
        """Q&Aペアの品質チェック"""
        MIN_QUESTION_LENGTH = 10
        MIN_ANSWER_LENGTH = 20
        MAX_ANSWER_LENGTH = 500

        if not all(key in qa_pair for key in ['question', 'answer']):
            return False

        question = qa_pair['question']
        answer = qa_pair['answer']

        if not question or not answer:
            return False

        if len(question) < MIN_QUESTION_LENGTH:
            return False
        if len(answer) < MIN_ANSWER_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
            return False
        if not question.endswith('？'):
            return False

        return True
```

主な機能：
- Q&Aペアの長さと形式の検証
- コンテンツの多様性チェック
- 重複検出

## データセット生成プロセス

### DatasetCreator クラス

全体のプロセスを管理し、データセットの生成を制御します。

```python
class DatasetCreator:
    def __init__(self):
        self.wiki_processor = WikiTextProcessor()
        self.qa_generator = QAGenerator()
        self.quality_checker = QualityChecker()
        
    def create_dataset(self, topics: List[str], output_file: str = "training_data.json") -> None:
        all_formatted_data = []
        
        for topic_idx, topic in enumerate(topics, 1):
            text = self.wiki_processor.get_wiki_text(topic)
            if not text:
                continue
                
            text = self.wiki_processor.clean_text(text)
            chunks = self.wiki_processor.split_into_chunks(text)
            
            for chunk in chunks:
                qa_pairs = self.qa_generator.generate_qa_pairs_with_retry(chunk)
                qa_pairs = [qa for qa in qa_pairs if self.quality_checker.validate_qa_pair(qa)]
                
                if self.quality_checker.check_diversity(qa_pairs):
                    formatted_data = self.qa_generator.format_for_llama(qa_pairs)
                    all_formatted_data.extend(formatted_data)
                    
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_formatted_data, f, ensure_ascii=False, indent=2)
```

## Hugging Faceへのデータセットアップロード

生成したデータセットをHugging Faceにアップロードする機能も実装されています。

```python
def create_and_upload_dataset(data_dict, dataset_name, username):
    dataset = Dataset.from_dict(data_dict)
    dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
    repo_id = f"{username}/{dataset_name}"
    
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
        )
        logger.success(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        logger.error(f"Error uploading dataset: {e}")
        raise
```

## 使用方法

1. 環境のセットアップ
```python
!pip install datasets huggingface-hub
```

2. トピックリストの定義
```python
topics = ["霊烏路空"]
```

3. データセット生成の実行
```python
creator = DatasetCreator()
creator.create_dataset(topics, "llama_training_data.json")
```

## おわりに

このツールを使用することで、Wikipediaの記事から高品質なQ&Aデータセットを自動生成し、Llama 3.1のファインチューニングに使用することができます。生成されたデータセットは自動的にHugging Faceにアップロードされ、共有や再利用が容易になっています。

## Google Colabノートブック

https://colab.research.google.com/drive/1mmK5vxUzjk3lI6OnEPrQqyjSzqsEoXpk?usp=sharing

## リポジトリ

https://github.com/Sunwood-ai-labs/Llama-finetune-sandbox
