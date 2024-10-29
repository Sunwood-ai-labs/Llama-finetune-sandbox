# LLM評価システム (LLMs as a Judge)

## はじめに

このノートブックでは、LLM（大規模言語モデル）の回答品質を自動的に評価するためのシステムを実装します。このシステムは、質問、模範解答、LLMの回答を比較し、4段階のスケールで評価を行います。

### 目的
- LLMの回答品質を定量的に評価する
- 評価プロセスを自動化し、大規模なデータセットの処理を可能にする
- 評価結果を分析可能な形式で出力する


### 評価基準
システムは以下の4段階スケールで評価を行います：
- **4点**: 優れた回答（完全で詳細な回答）
- **3点**: おおむね役立つ回答（改善の余地あり）
- **2点**: あまり役に立たない回答（重要な側面の欠落）
- **1点**: 全く役に立たない回答（無関係または不十分）

### 必要要件
- Python 3.7以上
- Google Colab環境
- Gemini API Key
- 評価対象のQAデータセット（JSON形式）

それでは、実装の詳細に進みましょう。

## 1. 環境セットアップ

必要なライブラリをインストールします。

```python
!pip install litellm tqdm loguru
```

必要なライブラリをインポートします。

```python
import json
import pandas as pd
from litellm import completion
import os
from tqdm import tqdm
import time
from google.colab import userdata
from loguru import logger
import sys
from functools import wraps
```

## 2. ユーティリティ関数の実装

### 2.1 リトライデコレータの実装

エラーハンドリングとリトライ機能を提供するデコレータクラスを実装します。

```python
def retry_on_error(max_retries=5, wait_time=30):
    """
    関数実行時のエラーを処理し、指定回数リトライするデコレータ
    
    Args:
        max_retries (int): 最大リトライ回数
        wait_time (int): リトライ間隔（秒）
    
    Returns:
        function: デコレートされた関数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"最大リトライ回数に達しました: {str(e)}")
                        raise
                    logger.warning(f"エラーが発生しました。{wait_time}秒後にリトライします。(試行 {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator
```

## 3. 評価システムのコアコンポーネント

### 3.1 プロンプト管理クラス

評価に使用するプロンプトを管理するクラスを実装します。

```python
class EvaluationPrompts:
    """評価プロンプトを管理するクラス"""
    
    @staticmethod
    def get_judge_prompt():
        return """
        あなたはLLMの回答を評価する審査員です。
        質問と模範解答、そしてLLMの回答のセットを評価してください。

        評価は1から4の整数スケールで行ってください：
        1: 全く役に立たない回答：質問に対して無関係か、部分的すぎる
        2: あまり役に立たない回答：質問の重要な側面を見落としている
        3: おおむね役立つ回答：支援を提供しているが、改善の余地がある
        4: 優れた回答：関連性があり、直接的で、詳細で、質問で提起されたすべての懸念に対応している

        以下のフォーマットで評価を提供してください：

        Feedback:::
        評価理由: (評価の根拠を説明してください)
        総合評価: (1から4の整数で評価してください)

        これから質問、模範解答、LLMの回答を提示します：

        質問: {question}
        模範解答: {correct_answer}
        LLMの回答: {llm_answer}

        フィードバックをお願いします。
        Feedback:::
        評価理由: """
```

### 3.2 評価結果パーサークラス

```python
class EvaluationParser:
    """評価結果を解析するクラス"""
    
    @staticmethod
    def extract_score(response_text):
        """
        評価テキストからスコアを抽出する
        
        Args:
            response_text (str): 評価テキスト
        
        Returns:
            int or None: 抽出されたスコア
        """
        try:
            score_text = response_text.split("総合評価:")[1].strip()
            score = int(score_text.split()[0])
            return score
        except:
            logger.error(f"スコア抽出に失敗しました: {response_text}")
            return None

    @staticmethod
    def extract_reason(evaluation_text):
        """
        評価テキストから評価理由を抽出する
        
        Args:
            evaluation_text (str): 評価テキスト
        
        Returns:
            str: 抽出された評価理由
        """
        try:
            reason = evaluation_text.split("評価理由:")[1].split("総合評価:")[0].strip()
            return reason
        except:
            logger.warning("評価理由の抽出に失敗しました")
            return ""
```

### 3.3 LLM評価クラス

```python
class LLMEvaluator:
    """LLMの回答を評価するメインクラス"""
    
    def __init__(self, model_name="gemini/gemini-pro"):
        """
        評価器を初期化する
        
        Args:
            model_name (str): 使用するLLMモデル名
        """
        self.model_name = model_name
        self.prompts = EvaluationPrompts()
        self.parser = EvaluationParser()
        logger.info(f"評価器を初期化しました。使用モデル: {model_name}")

    @retry_on_error()
    def evaluate_response(self, question, correct_answer, llm_answer):
        """
        個々の回答を評価する
        
        Args:
            question (str): 質問
            correct_answer (str): 模範解答
            llm_answer (str): LLMの回答
            
        Returns:
            dict: 評価結果
        """
        prompt = self.prompts.get_judge_prompt().format(
            question=question,
            correct_answer=correct_answer,
            llm_answer=llm_answer
        )
        
        try:
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            evaluation = response.choices[0].message.content
            score = self.parser.extract_score(evaluation)
            
            if score:
                logger.debug(f"評価完了 - スコア: {score}")
            
            return {
                'score': score,
                'evaluation': evaluation
            }
        except Exception as e:
            logger.error(f"評価中にエラーが発生しました: {str(e)}")
            raise

    @retry_on_error()
    def evaluate_dataset(self, json_file_path, output_file="evaluation_results.json"):
        """
        データセット全体を評価する

        Args:
            json_file_path (str): 評価対象のJSONファイルパス
            output_file (str): 評価結果の出力先ファイルパス

        Returns:
            dict: 評価結果と分析データを含む辞書
        """
        logger.info(f"データセット評価を開始します: {json_file_path}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        results = []
        qa_pairs = data['qa_pairs_simple']
        total_pairs = len(qa_pairs)
        
        logger.info(f"合計 {total_pairs} 件のQAペアを評価します")
        
        progress_bar = tqdm(qa_pairs, desc="評価進捗", unit="件")
        for qa in progress_bar:
            eval_result = self.evaluate_response(
                qa['question'],
                qa['correct_answer'],
                qa['llm_answer']
            )
            
            if eval_result:
                results.append({
                    'question': qa['question'],
                    'correct_answer': qa['correct_answer'],
                    'llm_answer': qa['llm_answer'],
                    'score': eval_result['score'],
                    'evaluation': eval_result['evaluation']
                })
                
                # 進捗状況を更新
                progress_bar.set_postfix(
                    completed=f"{len(results)}/{total_pairs}",
                    last_score=eval_result['score']
                )
            
            time.sleep(1)  # API制限考慮

        # 結果を分析
        scores = [r['score'] for r in results if r['score'] is not None]
        analysis = {
            'total_evaluations': len(results),
            'average_score': sum(scores) / len(scores) if scores else 0,
            'score_distribution': {
                '1': scores.count(1),
                '2': scores.count(2),
                '3': scores.count(3),
                '4': scores.count(4)
            }
        }

        # 分析結果をログに出力
        logger.success("評価が完了しました")
        logger.info(f"総評価数: {analysis['total_evaluations']}")
        logger.info(f"平均スコア: {analysis['average_score']:.2f}")
        logger.info("スコア分布:")
        for score, count in analysis['score_distribution'].items():
            percentage = (count / len(scores) * 100) if scores else 0
            logger.info(f"スコア {score}: {count}件 ({percentage:.1f}%)")

        # 結果をJSONとして保存
        output_data = {
            'analysis': analysis,
            'detailed_results': results
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"評価結果を保存しました: {output_file}")
        return output_data
```

### 3.4 データエクスポートクラス

```python
class ResultExporter:
    """評価結果をエクスポートするクラス"""
    
    @staticmethod
    def export_to_csv(evaluation_results, output_file="evaluation_results.csv"):
        """
        評価結果をCSVファイルに出力する
        
        Args:
            evaluation_results (dict): 評価結果
            output_file (str): 出力ファイルパス
        
        Returns:
            pd.DataFrame: 出力したデータフレーム
        """
        logger.info("CSV出力を開始します")
        results = evaluation_results['detailed_results']
        parser = EvaluationParser()
        
        csv_data = []
        for result in results:
            csv_data.append({
                '質問': result['question'],
                '正解': result['correct_answer'],
                'LLMの回答': result['llm_answer'],
                '評価理由': parser.extract_reason(result['evaluation']),
                '総合評価': result['score']
            })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        logger.success(f"CSVファイルを出力しました: {output_file}")
        return df
```

### 3.5 レポート生成クラス

```python
class ReportGenerator:
    """評価レポートを生成するクラス"""
    
    @staticmethod
    def generate_html_report(evaluation_results, model_name, output_file="evaluation_report.html"):
        """
        HTML形式の評価レポートを生成する
        
        Args:
            evaluation_results (dict): 評価結果
            model_name (str): 評価に使用したモデル名
            output_file (str): 出力ファイルパス
        """
        logger.info("HTMLレポート生成を開始します")
        
        analysis = evaluation_results['analysis']
        results = evaluation_results['detailed_results']
        df = pd.DataFrame(results)
        
        html_content = f"""
        <html>
        <head>
            <title>LLM Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background-color: #f0f0f0; padding: 20px; margin-bottom: 20px; }}
                .distribution {{ margin-bottom: 20px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>LLM Evaluation Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Total Evaluations: {analysis['total_evaluations']}</p>
                <p>Average Score: {analysis['average_score']:.2f}</p>
                <p>Model: {model_name}</p>
            </div>
            
            <div class="distribution">
                <h2>Score Distribution</h2>
                <table>
                    <tr>
                        <th>Score</th>
                        <th>Count</th>
                        <th>Percentage</th>
                    </tr>
                    {''.join(f'<tr><td>{score}</td><td>{count}</td><td>{(count/analysis["total_evaluations"]*100):.1f}%</td></tr>' 
                            for score, count in analysis['score_distribution'].items())}
                </table>
            </div>
            
            <div class="details">
                <h2>Detailed Results</h2>
                {df.to_html()}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.success(f"HTMLレポートを生成しました: {output_file}")
```

## 4. メイン実行部分

```python
def main():
    # APIキーの設定
    os.environ['GEMINI_API_KEY'] = userdata.get('GEMINI_API_KEY')

    # 評価器の初期化
    evaluator = LLMEvaluator(model_name="gemini/gemini-1.5-flash-latest")
    
    try:
        # データセットを評価
        logger.info("評価プロセスを開始します")
        results = evaluator.evaluate_dataset("qa_with_llm.json")
        
        # 結果のエクスポート
        exporter = ResultExporter()
        df = exporter.export_to_csv(results)
        logger.info("最初の数行のデータ:")
        logger.info("\n" + str(df.head()))
        
        # レポート生成
        report_generator = ReportGenerator()
        report_generator.generate_html_report(results, evaluator.model_name)
        logger.success("すべての処理が完了しました")
        
    except Exception as e:
        logger.error(f"処理中にエラーが発生しました: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

## 5. 使用方法

1. Google Colabで新しいノートブックを作成します。
2. 必要なライブラリをインストールします。
3. 上記のコードを順番にセルにコピーして実行します。
4. GEMINI_API_KEYを設定します。
5. 評価したいQAデータセットのJSONファイルを用意します。
6. メイン実行部分を実行します。

## 6. 注意点

- 評価には時間がかかる場合があります。
- API制限に注意してください。
- データセットは指定のJSON形式に従う必要があります。
- エラー発生時は自動的にリトライします。

