# コンテキストアウェアリフレクティブQA生成システム

このノートブックでは、Wikipediaの記事から高品質なQ&Aデータセットを生成するシステムを実装します。

## 1. 環境セットップ

```python
!curl https://ollama.ai/install.sh | sh

!echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections
!sudo apt-get update && sudo apt-get install -y cuda-drivers
```

```python
!nohup ollama serve &
```

```python
!ollama pull llama3.1:8b-instruct-fp16
```

```python
!pip install -q litellm tqdm loguru wikipedia transformers
!pip install -q datasets huggingface-hub
```

```python
import wikipedia
import json
from typing import List, Dict, Any
from loguru import logger
import re
from tqdm import tqdm

from google.colab import userdata
import os
```

```python

```

## 2. 基本設定

```python
# モデル設定
MODEL_NAME = "ollama/llama3.1:8b-instruct-fp16"
# MODEL_NAME = "groq/llama3-8b-8192"
# MODEL_NAME = "gemini/gemini-1.5-flash-latest"
# MODEL_NAME = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# 基本パラメータ
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP_SIZE = 500
DEFAULT_QA_PAIRS_PER_CHUNK = 5
API_BASE = "http://localhost:11434"

# Groqのセットアップ
# os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
# os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
# os.environ["TOGETHERAI_API_KEY"] = userdata.get('TOGETHERAI_API_KEY')
```

```python
from litellm import completion

response = completion(
    model=MODEL_NAME,
    messages=[{ "content": "東方地霊殿について教えて","role": "user"}],
    api_base=API_BASE
)
print(response)
```

```python
# import os
# from litellm import completion

# user_message = "Hello, whats the weather in San Francisco??"
# messages = [{ "content": user_message,"role": "user"}]
# model_name = "together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
# response = completion(model=model_name, messages=messages)
# print(response)
```

```python
import re
import json
from typing import Optional, Any, Dict
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from litellm import completion

import time



class JSONExtractor:
    """LLMの出力からJSONを抽出するクラス"""

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """テキストからJSONを抽出してパース"""
        try:
            # 最初の { から最後の } までを抽出
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end + 1]
                return json.loads(json_str)
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {str(e)}")
            return None

```

## 3. WikiTextProcessor クラスの実装

```python
class WikiTextProcessor:
    """Wikipediaテキストの処理とチャンク分割を行うクラス"""

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
        text = re.sub(r'\[\d+\]', '', text)  # 参照記号の削除
        text = re.sub(r'\n\s*\n', '\n', text)  # 余分な改行の削除
        return text.strip()

    @staticmethod
    def generate_summary(text: str) -> str:
        """テキスト全体のサマリーを生成"""
        from litellm import completion

        summary_prompt = f"""
        以下のテキストの重要なポイントを3-5行で要約してください。
        重要な固有名詞や数値は必ず含めてください。

        テキスト:
        {text}
        """

        response = completion(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=200,
            api_base=API_BASE
        )

        time.sleep(1)

        return response.choices[0].message.content

    @staticmethod
    def split_into_chunks_with_context(
        text: str,
        summary: str,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        overlap_size: int = DEFAULT_OVERLAP_SIZE
    ) -> List[Dict[str, str]]:
        """テキストを文脈情報とオーバーラップ情報付きでチャンク分割"""
        sentences = text.split('。')
        chunks = []
        current_chunk = []
        current_size = 0
        previous_text = ""  # 前のチャンクのテキストを保持

        # 平均文長を計算
        avg_sentence_length = sum(len(s) + 1 for s in sentences) / len(sentences)
        overlap_sentences = max(1, int(overlap_size / avg_sentence_length))

        for i, sentence in enumerate(sentences):
            sentence_size = len(sentence) + 1

            if current_size + sentence_size > max_chunk_size and current_chunk:
                # チャンクテキストの作成
                chunk_text = '。'.join(current_chunk) + '。'

                # チャンクを保存（オーバーラップテキストも含める）
                chunks.append({
                    'chunk_text': chunk_text,
                    'overlap_text': previous_text,  # 前のチャンクの最後の部分
                    'summary': summary,
                    'position': len(chunks) / (len(text) / max_chunk_size)  # 概算位置
                })

                # 次のチャンクのためにオーバーラップを準備
                previous_text = '。'.join(current_chunk[-overlap_sentences:]) + '。'

                # オーバーラップを考慮して新しいチャンクを開始
                current_chunk = current_chunk[-overlap_sentences:]
                current_size = sum(len(s) + 1 for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        # 最後のチャンクを追加
        if current_chunk:
            chunk_text = '。'.join(current_chunk) + '。'
            chunks.append({
                'chunk_text': chunk_text,
                'overlap_text': previous_text,
                'summary': summary,
                'position': 1.0
            })

        # 先頭チャンクのoverlap_textを調整（前方テキストがないため）
        if chunks:
            chunks[0]['overlap_text'] = '（先頭のため、オーバーラップテキストなし）'

        return chunks

    @staticmethod
    def get_next_chunk_preview(
        sentences: List[str],
        current_position: int,
        preview_sentences: int = 2
    ) -> str:
        """次のチャンクの冒頭部分を取得"""
        if current_position + preview_sentences >= len(sentences):
            return ""
        preview = sentences[current_position:current_position + preview_sentences]
        return '。'.join(preview) + '。'
```

## 4. QAGenerator クラスの実装

```python
class QAGenerator2:
    """Q&Aペアの生成を担当するクラス"""

    @staticmethod
    def generate_qa_pairs_with_context(
        chunk_data: Dict[str, str],
        num_pairs: int = DEFAULT_QA_PAIRS_PER_CHUNK,
        max_retries: int = 3
    ) -> List[Dict[str, str]]:
        """文脈を考慮してQ&Aペアを生成"""
        from litellm import completion

        prompt = f"""
以下のテキストから質問と回答のペアを{num_pairs}つ生成してください。
質問は必ずメインテキストの内容から作成し、補足情報は質問を明確にするためだけに使用してください。

## 全体の文脈（参考情報）:
{chunk_data['summary']}

## テキストの位置（参考情報）:
テキスト全体の{int(chunk_data['position'] * 100)}%付近

## オーバーラップ部分（参考情報）:
{chunk_data.get('overlap_text', '（オーバーラップ部分なし）')}

## メインテキスト（このテキストから質問を作成）:
{chunk_data['chunk_text']}

以下の条件をすべて満たすJSONを出力してください：

1. 質問生成のルール:
   - メインテキストの内容から質問を作成
   - 質問文だけで回答が一意に決まるように具体的に作成
   - 必要に応じて、文脈やオーバーラップ部分の情報を質問文に含めて明確化
   - 例: 「このキャラクターは何をしましたか？」(×) → 「霊烏路空は地霊殿でどのような役割を担っていましたか？」(○)

2. 回答生成のルール:
   - メインテキストを主な情報源として使用
   - 補足情報は回答の正確性を高めるためにのみ使用
   - 500文字以下で簡潔に記述

3. フォーマットのルール:
   - 厳密なJSON形式で出力（最後の要素にカンマをつけない）
   - すべての質問は日本語で記述
   - すべての質問は「？」で終わる

出力形式:
{
    "qa_pairs": [
        {"question": "具体的な質問1？", "answer": "メインテキストに基づく回答1"},
        {"question": "具体的な質問2？", "answer": "メインテキストに基づく回答2"}
    ]
}
"""

        for attempt in range(max_retries):
            try:

                response = completion(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7,
                    api_base=API_BASE
                )
                # response = completion(
                #     model=MODEL_NAME,
                #     messages=[{"role": "user", "content": prompt}],
                #     max_tokens=1000,
                #     temperature=0.7,
                #     api_base="http://localhost:11434"
                # )
                # response = completion(
                #     model=MODEL_NAME,
                #     messages=[{"role": "user", "content": prompt}],
                #     max_tokens=1000
                # )

                time.sleep(1)
                result = json.loads(response.choices[0].message.content)
                return result['qa_pairs']

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error("All attempts failed")
                    return []
```

```python

class QAGenerator:
    """Q&Aペアの生成を担当するクラス"""

    @staticmethod
    def generate_qa_pairs_with_context(
        chunk_data: Dict[str, str],
        num_pairs: int = DEFAULT_QA_PAIRS_PER_CHUNK,
        max_retries: int = 3
    ) -> List[Dict[str, str]]:
        """文脈を考慮してQ&Aペアを生成"""
        from litellm import completion

        json_format = '''
{
    "qa_pairs": [
        {"question": "東方地霊殿において霊烏路空はどのような役割を担っていましたか？", "answer": "霊烏路空は地霊殿の管理者として働いており、地下の秩序を維持する役割を担っていました。"},
        {"question": "東方地霊殿の開発元である上海アリス幻樂団はどのような組織ですか？", "answer": "上海アリス幻樂団は、東方Projectシリーズを開発している同人サークルです。"}
    ]
}
'''

        prompt = f"""
以下のテキストから質問と回答のペアを{num_pairs}つ生成してください。
質問は必ずメインテキストの内容から作成し、補足情報は質問を明確にするためだけに使用してください。

## 全体の文脈（参考情報）:
{chunk_data['summary']}

## テキストの位置（参考情報）:
テキスト全体の{int(chunk_data['position'] * 100)}%付近

## オーバーラップ部分（参考情報）:
{chunk_data.get('overlap_text', '（オーバーラップ部分なし）')}

## メインテキスト（このテキストから質問を作成）:
{chunk_data['chunk_text']}

以下の条件をすべて満たすJSONを出力してください：

1. 質問生成の必須ルール:
   - メインテキストの内容から質問を作成すること
   - 各質問は完全に独立して理解可能であること
   - 固有名詞を明示的に含めること
   - 「この」「その」などの指示語を使用しないこと
   - 「彼」「彼女」などの代名詞を使用しないこと
   - 質問に登場する対象を常に具体的に明示すること

2. 質問作成の禁止事項:
   × 「このキャラクターは何をしましたか？」
   × 「彼女の役割は何ですか？」
   × 「その時何が起こりましたか？」
   × 「ここで何が行われましたか？」

3. 質問作成の好例:
   ○ 「霊烏路空は地霊殿でどのような役割を担っていましたか？」
   ○ 「東方地霊殿のストーリーで温泉はどのような意味を持っていましたか？」
   ○ 「上海アリス幻樂団が開発した東方地霊殿の特徴は何ですか？」

4. 回答生成のルール:
   - メインテキストを主な情報源として使用する
   - 補足情報は回答の正確性を高めるためにのみ使用する
   - 500文字以下で簡潔に記述する
   - 回答も代名詞を避け、具体的な固有名詞を使用する

5. フォーマットのルール:
   - 厳密なJSON形式で出力すること
   - 最後の要素にカンマをつけないこと
   - すべての質問は日本語で記述すること
   - すべての質問は「？」で終わること

出力形式:
{json_format}"""

        json_extractor = JSONExtractor()

        for attempt in range(max_retries):

            try:
                response = completion(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7,
                    api_base=API_BASE
                )
                time.sleep(1)

                # response = completion(
                #     model=MODEL_NAME,
                #     messages=[{"role": "user", "content": prompt}],
                #     max_tokens=1000,
                #     temperature=0.7,
                #     api_base="http://localhost:11434"
                # )
                # response = completion(
                #     model=MODEL_NAME,
                #     messages=[{"role": "user", "content": prompt}],
                #     max_tokens=1000
                # )

                # LLMの応答を取得
                llm_response = response.choices[0].message.content
                logger.debug(f"LLM Response (Attempt {attempt + 1}):\n{llm_response}")

                # JSONの抽出を試みる
                result = json_extractor.extract_json(llm_response)

                if result and 'qa_pairs' in result:
                    return result['qa_pairs']

                logger.warning(f"Failed to extract valid JSON (Attempt {attempt + 1})")
                logger.warning(f"Raw response:\n{llm_response}")

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error("All attempts failed")
                    logger.error("Last LLM response:\n%s", response.choices[0].message.content if 'response' in locals() else "No response")
                    return []

        return []


    @staticmethod
    def format_for_llama(qa_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Q&AペアをLlama 3.2のフォーマットに変換"""
        formatted_data = []
        for qa in qa_pairs:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": qa['question']},
                {"role": "assistant", "content": qa['answer']}
            ]
            formatted_data.append({"conversations": conversation})
        return formatted_data

```

```python
from dataclasses import dataclass
```

```python


@dataclass
class QAEvaluation:
    """Q&Aペアの評価結果を保持するデータクラス"""
    score: float
    feedback: str
    improvement_suggestions: List[str]
    factuality_score: float  # 事実との一致度
    question_quality_score: float  # 質問の質スコア
    answer_completeness_score: float  # 回答の完全性スコア

@dataclass
class QAImprovement:
    """Q&Aペアの改善プロセスを記録するデータクラス"""
    original_question: str
    original_answer: str
    improved_question: str
    improved_answer: str
    chunk_text: str
    chunk_position: float
    summary: str
    evaluation: QAEvaluation
    improvement_count: int
    topic: str
    timestamp: datetime



@dataclass
class QAEvaluation:
    """Q&Aペアの評価結果を保持するデータクラス"""
    score: float
    feedback: str
    improvement_suggestions: List[str]
    factuality_score: float  # 事実との一致度
    question_quality_score: float  # 質問の質スコア
    answer_completeness_score: float  # 回答の完全性スコア

@dataclass
class QAImprovement:
    """Q&Aペアの改善プロセスを記録するデータクラス"""
    original_question: str
    original_answer: str
    improved_question: str
    improved_answer: str
    chunk_text: str
    chunk_position: float
    summary: str
    evaluation: QAEvaluation
    improvement_count: int
    topic: str
    timestamp: datetime

class ReflexiveQAGenerator:
    """リフレクションベースのQA生成・評価・記録システム"""

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.improvements: List[QAImprovement] = []
        self.setup_output_directories()
        self.json_extractor = JSONExtractor()

    def setup_output_directories(self):
        """出力ディレクトリの設定"""
        self.output_dir = Path("qa_generation_output")
        self.output_dir.mkdir(exist_ok=True)
        self.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_evaluation_with_retry(self, qa_pair: Dict[str, str], context: Dict[str, str]) -> QAEvaluation:
        """評価を取得する（リトライ機能付き）"""
        prompt = self._create_evaluation_prompt(qa_pair, context)

        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            api_base=API_BASE
        )
        time.sleep(1)

        result = self.json_extractor.extract_json(response.choices[0].message.content)
        if not result:
            raise ValueError("有効なJSONが抽出できませんでした")

        try:
            return QAEvaluation(**result)
        except Exception as e:
            logger.error(f"QAEvaluation生成エラー: {str(e)}")
            raise ValueError("QAEvaluation生成に失敗しました")

    def evaluate_qa_pair(self, qa_pair: Dict[str, str], context: Dict[str, str]) -> QAEvaluation:
        """Q&Aペアを評価し、詳細なフィードバックを生成"""
        try:
            return self._get_evaluation_with_retry(qa_pair, context)
        except Exception as e:
            logger.error(f"評価に完全に失敗: {str(e)}")
            return QAEvaluation(
                score=0.0,
                factuality_score=0.0,
                question_quality_score=0.0,
                answer_completeness_score=0.0,
                feedback="評価処理中に重大なエラーが発生しました",
                improvement_suggestions=[]
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _get_llm_response_with_retry(self, prompt: str) -> str:
        """LLMからのレスポンス取得（リトライ付き）"""
        response = completion(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            api_base=API_BASE
        )
        time.sleep(1)

        result = self.json_extractor.extract_json(response.choices[0].message.content)
        if not result:
            raise ValueError(f"Invalid JSON in LLM response: {response.choices[0].message.content[:200]}...")

        if not all(k in result for k in ['question', 'answer']):
            raise ValueError(f"Missing required keys in JSON: {result}")

        return result

    def _improve_qa_pair_with_retry(
        self,
        qa_pair: Dict[str, str],
        evaluation: QAEvaluation,
        context: Dict[str, str]
    ) -> Dict[str, str]:
        """Q&Aペアの改善を試みる"""
        prompt = self._create_improvement_prompt(qa_pair, evaluation, context)

        try:
            return self._get_llm_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"Failed to improve QA pair after all retries: {str(e)}")
            return qa_pair  # 失敗した場合は元のQ&Aペアを返す



    def improve_qa_pair(
        self,
        qa_pair: Dict[str, str],
        evaluation: QAEvaluation,
        context: Dict[str, str]
    ) -> Dict[str, str]:
        """評価結果に基づいてQ&Aペアを改善"""
        try:
            return self._improve_qa_pair_with_retry(qa_pair, evaluation, context)
        except Exception as e:
            logger.error(f"改善に完全に失敗: {str(e)}")
            return qa_pair

    def _create_evaluation_prompt(self, qa_pair: Dict[str, str], context: Dict[str, str]) -> str:
        """評価用のプロンプトを生成"""
        return f"""
以下のQ&Aペアの品質を厳密に評価してください。

## 記事の要約:
{context['summary']}

## 現在のチャンク（テキスト全体の{int(context['position'] * 100)}%付近）:
{context['chunk_text']}

## 評価対象:
質問: {qa_pair['question']}
回答: {qa_pair['answer']}

以下の基準で評価し、JSON形式で出力してください:
改善提案は内容を正確に考慮した改善提案をしてください。

1. 事実性 (factuality_score):
   - 回答は与えられたコンテキストと完全に一致しているか
   - サマリーの内容とも矛盾していないか
   - スコアは0.0～1.0で評価

2. 質問の質 (question_quality_score):
   - 指示語や代名詞を避けているか
   - 具体的な固有名詞を使用しているか
   - 質問は文脈なしで理解可能か
   - スコアは0.0～1.0で評価

3. 回答の完全性 (answer_completeness_score):
   - 回答は質問に対して適切で完全な情報を提供しているか
   - 必要な文脈や詳細が含まれているか
   - スコアは0.0～1.0で評価

必ず以下の形式でJSONを出力してください:
{{
    "factuality_score": 0.0～1.0の評価スコア,
    "question_quality_score": 0.0～1.0の評価スコア,
    "answer_completeness_score": 0.0～1.0の評価スコア,
    "score": 3つのスコアの平均値,
    "feedback": "詳細な評価コメント",
    "improvement_suggestions": [
        "改善提案1",
        "改善提案2"
    ]
}}

注意: 必ず有効なJSONを出力してください。コードブロックや追加の説明は不要です。"""

    def _create_improvement_prompt(
        self,
        qa_pair: Dict[str, str],
        evaluation: QAEvaluation,
        context: Dict[str, str]
    ) -> str:
        """改善用のプロンプトを生成"""
        return f"""
Q&Aペアを以下の情報に基づいて改善してください。

## 記事の要約:
{context['summary']}

## 現在のチャンク（テキスト全体の{int(context['position'] * 100)}%付近）:
{context['chunk_text']}

## 現在のQ&Aペア:
質問: {qa_pair['question']}
回答: {qa_pair['answer']}

## 評価スコア:
- 事実との一致度: {evaluation.factuality_score}
- 質問の質: {evaluation.question_quality_score}
- 回答の完全性: {evaluation.answer_completeness_score}

## 評価フィードバック:
{evaluation.feedback}

## 改善提案:
{json.dumps(evaluation.improvement_suggestions, ensure_ascii=False, indent=2)}

以下の条件を満たす改善されたQ&Aペアを生成してください:
1. 与えられたコンテキストの内容に完全に即した事実のみを含める
2. 指示語や代名詞を避け、具体的な固有名詞を使用する
3. 質問と回答は独立して理解可能にする
4. サマリー情報も参考に、より正確な文脈を提供する

必ず以下の形式でJSONを出力してください:
{{
    "question": "改善された質問",
    "answer": "改善された回答"
}}

注意: 必ず有効なJSONを出力してください。コードブロックや追加の説明は不要です。"""

    def generate_qa_pairs_with_reflection(
        self,
        chunk_data: Dict[str, str],
        topic: str,
        num_pairs: int = 3,
        quality_threshold: float = 0.8,
        max_improvement_attempts: int = 2
    ) -> List[Dict[str, str]]:
        """リフレクションを用いて高品質なQ&Aペアを生成"""
        qa_generator = QAGenerator()
        final_qa_pairs = []

        # チャンク情報のログ出力
        logger.info(f"=== チャンク情報 ===")
        logger.info(f"位置: テキスト全体の{int(chunk_data['position'] * 100)}%付近")
        logger.info(f"サマリー: {chunk_data['summary'][:100]}...")

        initial_pairs = qa_generator.generate_qa_pairs_with_context(chunk_data, num_pairs)
        logger.info(f"初期Q&Aペア数: {len(initial_pairs)}")

        for pair_idx, qa_pair in enumerate(initial_pairs, 1):
            current_pair = qa_pair
            best_evaluation = None
            improvement_count = 0

            logger.info(f"--- Q&Aペア {pair_idx}/{len(initial_pairs)} の改善プロセス ---")
            logger.info("初期Q&A:")
            logger.info(f"Q: {qa_pair['question']}")
            logger.info(f"A: {qa_pair['answer']}\n")

            for attempt in range(max_improvement_attempts):
                evaluation = self.evaluate_qa_pair(current_pair, chunk_data)

                # 評価結果の詳細表示
                logger.info(f"改善試行 {attempt + 1}/{max_improvement_attempts}")
                logger.info(f"評価スコア:")
                logger.info(f"- 事実との一致度: {evaluation.factuality_score:.2f}")
                logger.info(f"- 質問の質: {evaluation.question_quality_score:.2f}")
                logger.info(f"- 回答の完全性: {evaluation.answer_completeness_score:.2f}")
                logger.info(f"- 総合スコア: {evaluation.score:.2f}")
                logger.info(f"フィードバック: {evaluation.feedback}")
                if evaluation.improvement_suggestions:
                    logger.info("改善提案:")
                    for i, suggestion in enumerate(evaluation.improvement_suggestions, 1):
                        logger.info(f"  {i}. {suggestion}")

                if not best_evaluation or evaluation.score > best_evaluation.score:
                    best_evaluation = evaluation
                    best_pair = current_pair
                    logger.info("✨ 新しいベストスコアを記録")

                if evaluation.score >= quality_threshold:
                    logger.info(f"✅ 品質閾値 {quality_threshold} を達成")
                    break

                current_pair = self.improve_qa_pair(current_pair, evaluation, chunk_data)
                if current_pair != qa_pair:  # 改善が行われた場合
                    logger.info("改善後のQ&A:")
                    logger.info(f"Q: {current_pair['question']}")
                    logger.info(f"A: {current_pair['answer']}")
                improvement_count += 1

            if best_evaluation and best_evaluation.score >= quality_threshold:
                final_qa_pairs.append(best_pair)
                logger.info(f"✅ Q&Aペア {pair_idx} を採用 (スコア: {best_evaluation.score:.2f})")

                # 改善履歴を記録
                improvement = QAImprovement(
                    original_question=qa_pair['question'],
                    original_answer=qa_pair['answer'],
                    improved_question=best_pair['question'],
                    improved_answer=best_pair['answer'],
                    chunk_text=chunk_data['chunk_text'],
                    chunk_position=chunk_data['position'],
                    summary=chunk_data['summary'],
                    evaluation=best_evaluation,
                    improvement_count=improvement_count,
                    topic=topic,
                    timestamp=datetime.now()
                )
                self.improvements.append(improvement)
            else:
                logger.warning(f"❌ Q&Aペア {pair_idx} は品質基準を満たさず不採用")

        logger.info(f"=== 最終結果 ===")
        logger.info(f"生成されたQ&Aペア数: {len(final_qa_pairs)}/{len(initial_pairs)}")
        if final_qa_pairs:
            avg_score = sum(imp.evaluation.score for imp in self.improvements[-len(final_qa_pairs):]) / len(final_qa_pairs)
            logger.info(f"平均品質スコア: {avg_score:.2f}")

        return final_qa_pairs

    def save_to_csv(self, topic: str = "unknown") -> tuple[Path, Path]:
        """改善履歴をCSVファイルに保存"""
        csv_path = self.output_dir / f"qa_improvements_{self.current_timestamp}.csv"

        records = []
        for imp in self.improvements:
            record = {
                "topic": imp.topic,
                "timestamp": imp.timestamp.isoformat(),
                "chunk_position": imp.chunk_position,
                "original_question": imp.original_question,
                "original_answer": imp.original_answer,
                "improved_question": imp.improved_question,
                "improved_answer": imp.improved_answer,
                "factuality_score": imp.evaluation.factuality_score,
                "question_quality_score": imp.evaluation.question_quality_score,
                "answer_completeness_score": imp.evaluation.answer_completeness_score,
                "overall_score": imp.evaluation.score,
                "feedback": imp.evaluation.feedback,
                "improvement_suggestions": "; ".join(imp.evaluation.improvement_suggestions),
                "improvement_count": imp.improvement_count,
                "chunk_text": imp.chunk_text,
                "summary": imp.summary
            }
            records.append(record)

        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"改善履歴をCSVに保存しました: {csv_path}")

        # 基本的な統計情報を出力
        stats = {
            "total_qa_pairs": len(records),
            "avg_improvement_count": df["improvement_count"].mean(),
            "avg_final_score": df["overall_score"].mean(),
            "improved_pairs": len(df[df["improvement_count"] > 0])
        }

        stats_path = self.output_dir / f"qa_stats_{self.current_timestamp}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return csv_path, stats_path

    def format_for_llama(self, qa_pairs: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Q&AペアをLlama形式に変換"""
        formatted_data = []
        for qa in qa_pairs:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": qa['question']},
                {"role": "assistant", "content": qa['answer']}
            ]
            formatted_data.append({"conversations": conversation})
        return formatted_data


class DatasetCreator:
    """データセット生成の全体プロセスを管理するクラス"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or DEFAULT_CONFIG
        self.wiki_processor = WikiTextProcessor()
        self.qa_generator = ReflexiveQAGenerator(model_name=self.config["model_name"])
        self.setup_directories()

    def setup_directories(self):
        """必要なディレクトリの作成"""
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

    def create_dataset(
        self,
        topics: List[str],
        output_file: str = "qa_dataset.json"
    ) -> None:
        """データセット生成のメインプロセス"""
        all_qa_pairs = []

        for topic_idx, topic in enumerate(topics, 1):
            logger.info(f"Processing topic {topic_idx}/{len(topics)}: {topic}")

            try:
                # Wikipedia記事の取得と前処理
                text = self.wiki_processor.get_wiki_text(topic)
                if not text:
                    logger.warning(f"No text found for topic: {topic}")
                    continue

                text = self.wiki_processor.clean_text(text)
                summary = self.wiki_processor.generate_summary(text)

                # チャンク分割
                chunks = self.wiki_processor.split_into_chunks_with_context(
                    text,
                    summary,
                    self.config["chunk_size"],
                    self.config["overlap_size"]
                )

                # chunks = chunks[:2]

                # 各チャンクからQ&Aペアを生成
                for chunk in tqdm(chunks, desc=f"Generating Q&A pairs for {topic}"):
                    qa_pairs = self.qa_generator.generate_qa_pairs_with_reflection(
                        chunk_data=chunk,
                        topic=topic,
                        num_pairs=self.config["qa_pairs_per_chunk"],
                        quality_threshold=self.config["quality_threshold"],
                        max_improvement_attempts=self.config["max_improvement_attempts"]
                    )

                    if qa_pairs:
                        formatted_pairs = self.qa_generator.format_for_llama(qa_pairs)
                        all_qa_pairs.extend(formatted_pairs)

            except Exception as e:
                logger.error(f"Error processing topic {topic}: {str(e)}")
                continue

        if all_qa_pairs:
            # データセットの保存
            output_path = output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)

            # 改善履歴と統計情報の保存
            csv_path, stats_path = self.qa_generator.save_to_csv()

            logger.info(f"""
生成結果を保存しました:
- データセット: {output_path}
- 改善履歴: {csv_path}
- 統計情報: {stats_path}
""")
        else:
            logger.warning("有効なQ&Aペアが生成されませんでした。")

```

## 5. QualityChecker クラスの実装

```python
class QualityChecker:
    """生成されたQ&Aペアの品質管理を行うクラス"""

    @staticmethod
    def validate_qa_pair(qa_pair: Dict[str, str]) -> bool:
        """Q&Aペアの品質チェック"""
        MIN_QUESTION_LENGTH = 10
        MIN_ANSWER_LENGTH = 20
        MAX_ANSWER_LENGTH = 500

        # 必須キーの存在チェック
        if not all(key in qa_pair for key in ['question', 'answer']):
            return False

        question = qa_pair['question']
        answer = qa_pair['answer']

        # 空文字チェック
        if not question or not answer:
            return False

        if len(question) < MIN_QUESTION_LENGTH:
            return False
        if len(answer) < MIN_ANSWER_LENGTH or len(answer) > MAX_ANSWER_LENGTH:
            return False
        if not question.endswith('？'):
            return False

        return True

    @staticmethod
    def check_diversity(qa_pairs: List[Dict[str, str]], threshold: float = 0.7) -> bool:
        """Q&Aペアの多様性をチェック"""
        # 無効なペアを除外
        valid_pairs = [pair for pair in qa_pairs if QualityChecker.validate_qa_pair(pair)]

        if not valid_pairs:  # 有効なペアが1つもない場合
            return False

        from difflib import SequenceMatcher

        for i, qa1 in enumerate(valid_pairs):
            for j, qa2 in enumerate(valid_pairs[i+1:]):
                similarity = SequenceMatcher(
                    None,
                    qa1['question'] + qa1['answer'],
                    qa2['question'] + qa2['answer']
                ).ratio()

                if similarity > threshold:
                    return False
        return True

```

## 6. DatasetCreator クラスの実装

```python

```

## 7. 使用例


## 8. Hugging Faceへのアップロード (オプション)


## 9. パラメータチューニングのヒント

- チャンクサイズ（`chunk_size`）:
  - 短すぎる: 文脈が失われる
  - 長すぎる: Q&Aペアが散漫になる
  - 推奨: 150-300文字

- オーバーラップサイズ（`overlap_size`）:
  - 小さすぎる: 文脈の連続性が失われる
  - 大きすぎる: 重複が多くなりすぎる
  - 推奨: チャンクサイズの40-60%

- Q&Aペア生成数（`num_pairs`）:
  - 少なすぎる: データセットが小さくなる
  - 多すぎる: 品質が低下する可能性
  - 推奨: チャンクあたり3-7ペア

```python
import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from loguru import logger
from typing import Dict, Any

class DatasetUploader:
    def __init__(self, username: str, dataset_name: str):
        self.username = username
        self.dataset_name = dataset_name
        self.repo_id = f"{username}/{dataset_name}"

    def load_data(self, file_path: str) -> Dict[str, Any]:
        """JSONファイルからデータを読み込む"""
        logger.info(f"Loading data from {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.success(f"Successfully loaded data from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise

    def process_data(self, data: Dict[str, Any]) -> Dict[str, list]:
        """データを処理して必要な形式に変換"""
        logger.info("Processing data")
        processed_data = {
            "instruction": [],
            "input": [],
            "output": [],
            "system": []
        }

        total_items = len(data)
        for idx, item in enumerate(data, 1):
            if idx % 1000 == 0:
                logger.info(f"Processing item {idx}/{total_items} ({(idx/total_items)*100:.1f}%)")

            conversations = item["conversations"]
            system_prompt = next((conv["content"] for conv in conversations if conv["role"] == "system"), "")
            user_input = next((conv["content"] for conv in conversations if conv["role"] == "user"), "")
            assistant_output = next((conv["content"] for conv in conversations if conv["role"] == "assistant"), "")

            processed_data["system"].append(system_prompt)
            processed_data["instruction"].append(user_input)
            processed_data["input"].append("")
            processed_data["output"].append(assistant_output)

        logger.success(f"Successfully processed {total_items} items")
        return processed_data

    def upload_to_hub(self, processed_data: Dict[str, list]) -> None:
        """データセットを作成してHugging Faceにアップロード"""
        logger.info("Creating dataset from processed data")
        try:
            # データセットの作成
            dataset = Dataset.from_dict(processed_data)
            logger.info(f"Created dataset with {len(dataset)} examples")

            # トレーニング/テストセットの分割
            dataset_dict = dataset.train_test_split(test_size=0.1, seed=42)
            logger.info(f"Train set: {len(dataset_dict['train'])} examples")
            logger.info(f"Test set: {len(dataset_dict['test'])} examples")

            # アップロード
            logger.info(f"Uploading dataset to {self.repo_id}")
            dataset_dict.push_to_hub(repo_id=self.repo_id)
            logger.success(f"Dataset uploaded successfully to: https://huggingface.co/datasets/{self.repo_id}")

        except Exception as e:
            logger.error(f"Error uploading dataset: {e}")
            raise
```

# メインスクリプトの実装

## 1. メイン実行ファイル（main.py）

```python
import json
from loguru import logger
from pathlib import Path
from typing import List
from datetime import datetime
from datasets import load_dataset

# 設定値を変数として定義
topic = "霊烏路空"
output_file = "qa_dataset.json"
chunk_size = 200
overlap_size = 700
qa_pairs = 5
hf_upload = True  # テスト用にTrueに設定
hf_username = "MakiAi"
hf_dataset_name = f"OKU_wiki_llama3.1_8b_inst_Reflexive_chunk{chunk_size}_overlap{overlap_size}"
log_dir = "logs"

# ReflexiveQAGeneratorの追加設定
quality_threshold = 0.8
max_improvement_attempts = 2
# model_name = "gpt-4"

def load_topics(file_path: str) -> List[str]:
    """トピックリストをファイルから読み込む"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def main():

    if Path(topic).exists():
        topics = load_topics(topic)
        logger.info(f"Loaded {len(topics)} topics from {topic}")
    else:
        topics = [topic]
        logger.info(f"Using single topic: {topic}")

    try:
        # データセット生成
        creator = DatasetCreator(config={
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "qa_pairs_per_chunk": qa_pairs,
            "quality_threshold": quality_threshold,
            "max_improvement_attempts": max_improvement_attempts,
            "model_name": MODEL_NAME
        })

        creator.create_dataset(
            topics=topics,
            output_file=output_file
        )

        # Hugging Faceへのアップロード
        if hf_upload:
            if not all([hf_username, hf_dataset_name]):
                logger.error("Hugging Face upload requires username and dataset name")
                return

            try:
                logger.info("Uploading dataset to Hugging Face Hub...")
                # DatasetUploaderのインスタンス化
                uploader = DatasetUploader(
                    username=hf_username,
                    dataset_name=hf_dataset_name
                )

                # データの読み込み
                data = uploader.load_data(output_file)

                # データの処理
                processed_data = uploader.process_data(data)

                # Hugging Faceへのアップロード
                uploader.upload_to_hub(processed_data)
                logger.success("Successfully uploaded dataset to Hugging Face Hub")
            except Exception as e:
                logger.error(f"Failed to upload to Hugging Face Hub: {str(e)}")

    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

## 2. 実行方法例

### 基本的な使用方法

```bash
# 単一トピックでの実行
# python main.py --topics "霊烏路空"

# トピックリストファイルからの実行
# python main.py --topics topics.txt --output utsuho_dataset.json

# パラメータのカスタマイズ
# python main.py \
#     --topics topics.txt \
#     --chunk-size 250 \
#     --overlap-size 120 \
#     --qa-pairs 7
```

### Hugging Faceへのアップロードを含む実行

```python
# !python main.py \
#     --topics "霊烏路空" \
#     --hf-upload \
#     --hf-username MakiAi \
#     --hf-dataset-name OKU_wiki_llama3.1_8b_inst
```

## 3. トピックリストファイル（topics.txt）の例

```text
霊烏路空
物部布都
四季映姫・ヤマザナドゥ
八雲紫
```

## 4. 実行結果の例

```text
2024-10-30 15:30:12 | INFO | Loaded 4 topics from topics.txt
2024-10-30 15:30:12 | INFO | Processing topic 1/4: 霊烏路空
2024-10-30 15:30:15 | INFO | Generated summary for 霊烏路空
2024-10-30 15:30:20 | INFO | Created 8 chunks with overlap
2024-10-30 15:31:05 | INFO | Generated 35 valid Q&A pairs
...
2024-10-30 15:45:23 | SUCCESS | Generated 156 Q&A pairs in total
2024-10-30 15:45:25 | SUCCESS | Dataset saved to touhou_qa.json
```
