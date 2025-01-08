from evaluate import load
import numpy as np
import torch
from typing import List, Dict, Union, Optional
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_rouge(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """ROUGE 점수 계산을 위한 헬퍼 함수"""
    rouge = load('rouge')
    results = rouge.compute(predictions=predictions, references=references)
    
    return {
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL']
    }

class SummarizationMetrics:
    """텍스트 생성/요약 평가를 위한 통합 메트릭 클래스"""
    
    def __init__(self, config):
        """
        Args:
            config: Hydra config object containing metrics settings
        """
        self.metrics = {}
        metrics_cfg = config.metrics
        
        # ROUGE 설정
        if metrics_cfg.rouge.enabled:
            self.metrics['rouge'] = load('rouge')
            self.rouge_config = metrics_cfg.rouge
            
        # BERTScore 설정
        if metrics_cfg.bertscore.enabled:
            self.metrics['bertscore'] = load('bertscore')
            self.bertscore_config = metrics_cfg.bertscore
            
        # METEOR 설정
        if metrics_cfg.meteor.enabled:
            self.metrics['meteor'] = load('meteor')
            
        # BLEURT 설정
        if metrics_cfg.bleurt.enabled:
            self.metrics['bleurt'] = load('bleurt')
            
        # BLEU 설정
        if metrics_cfg.bleu.enabled:
            self.bleu_smoothing = SmoothingFunction()
            smooth_method = getattr(
                self.bleu_smoothing, 
                metrics_cfg.bleu.smooth_method
            )
            self.bleu_config = {
                'smoothing_function': smooth_method
            }

    def compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        batch_size: Optional[int] = None,
        metrics_to_compute: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        모든 메트릭을 계산하고 결과를 반환
        
        Args:
            predictions: 모델이 생성한 텍스트 리스트
            references: 정답 텍스트 리스트
            batch_size: 로깅을 위한 배치 크기 (선택사항)
            metrics_to_compute: 계산할 특정 메트릭 리스트 (선택사항)
        
        Returns:
            Dict[str, float]: 메트릭 이름과 점수를 포함하는 딕셔너리
        """
        results = {}
        
        # ROUGE 스코어 계산
        if 'rouge' in self.metrics and (metrics_to_compute is None or 'rouge' in metrics_to_compute):
            rouge_scores = self._compute_rouge(predictions, references)
            results.update(rouge_scores)
            
        # BERTScore 계산
        if 'bertscore' in self.metrics and (metrics_to_compute is None or 'bertscore' in metrics_to_compute):
            bert_scores = self._compute_bertscore(predictions, references)
            results.update(bert_scores)
            
        # METEOR 계산
        if 'meteor' in self.metrics and (metrics_to_compute is None or 'meteor' in metrics_to_compute):
            meteor_score = self._compute_meteor(predictions, references)
            results['meteor'] = meteor_score
            
        # BLEU 계산
        if 'bleu' in self.metrics and (metrics_to_compute is None or 'bleu' in metrics_to_compute):
            bleu_scores = self._compute_bleu(predictions, references)
            results.update(bleu_scores)
            
        # BLEURT 계산
        if 'bleurt' in self.metrics and (metrics_to_compute is None or 'bleurt' in metrics_to_compute):
            bleurt_score = self._compute_bleurt(predictions, references)
            results['bleurt'] = bleurt_score
            
        return results

    def _compute_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """상세 ROUGE 스코어 계산"""
        results = self.metrics['rouge'].compute(
            predictions=predictions,
            references=references,
            use_stemmer=self.rouge_config.use_stemmer,
            use_aggregator=self.rouge_config.use_aggregator
        )
        
        # 각 ROUGE 점수의 ��균값 계산
        return {
            'rouge1': float(np.mean(results['rouge1'])),
            'rouge2': float(np.mean(results['rouge2'])),
            'rougeL': float(np.mean(results['rougeL']))
        }

    def _compute_bertscore(
        self, 
        predictions: List[str], 
        references: List[str],
        model_type: str = "microsoft/deberta-xlarge-mnli"
    ) -> Dict[str, float]:
        """BERTScore 계산"""
        results = self.metrics['bertscore'].compute(
            predictions=predictions,
            references=references,
            model_type=model_type,
            lang="en"
        )
        
        return {
            'bertscore_precision': np.mean(results['precision']),
            'bertscore_recall': np.mean(results['recall']),
            'bertscore_f1': np.mean(results['f1'])
        }

    def _compute_meteor(self, predictions: List[str], references: List[str]) -> float:
        """METEOR 스코어 계산"""
        return self.metrics['meteor'].compute(
            predictions=predictions,
            references=references
        )['meteor']

    def _compute_bleu(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """BLEU 스코어 계산 (1-4gram)"""
        bleu_scores = {}
        
        for i in range(1, 5):
            weights = tuple([1.0/i]*i + [0]*(4-i))
            scores = []
            
            for pred, ref in zip(predictions, references):
                score = sentence_bleu(
                    [ref.split()],
                    pred.split(),
                    weights=weights,
                    smoothing_function=self.bleu_smoothing
                )
                scores.append(score)
                
            bleu_scores[f'bleu_{i}'] = np.mean(scores)
            
        return bleu_scores

    def _compute_bleurt(self, predictions: List[str], references: List[str]) -> float:
        """BLEURT 스코어 계산"""
        results = self.metrics['bleurt'].compute(
            predictions=predictions,
            references=references
        )
        return np.mean(results['scores'])
