from evaluate import load
import evaluate
from typing import List, Dict, Union, Optional
import numpy as np
import logging

# absl 로거 끄기
logging.getLogger('absl').setLevel(logging.ERROR)

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE 스코어 계산"""
    try:
        # 새로운 방식으로 ROUGE 로드 시도
        rouge = evaluate.load('rouge', download_config=evaluate.DownloadConfig(token=True))
    except:
        try:
            # 이전 방식으로 시도
            rouge = load('rouge', token=True)
        except:
            # 마지막 대안으로 직접 ROUGE 계산
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            
            scores = {
                'rouge1': [],
                'rouge2': [],
                'rougeL': []
            }
            
            for pred, ref in zip(predictions, references):
                score = scorer.score(ref, pred)
                scores['rouge1'].append(score['rouge1'].fmeasure)
                scores['rouge2'].append(score['rouge2'].fmeasure)
                scores['rougeL'].append(score['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(scores['rouge1']),
                'rouge2': np.mean(scores['rouge2']),
                'rougeL': np.mean(scores['rougeL'])
            }
    
    # ROUGE 계산
    results = rouge.compute(predictions=predictions, references=references)
    
    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL']
    }

class SummarizationMetrics:
    def __init__(self, config):
        self.config = config
    
    def compute_metrics(self, 
                       predictions: List[str], 
                       references: List[str]) -> Dict[str, float]:
        """모든 메트릭 계산"""
        metrics = {}
        
        # ROUGE 스코어 계산
        rouge_scores = compute_rouge(predictions, references)
        metrics.update(rouge_scores)
        
        return metrics
