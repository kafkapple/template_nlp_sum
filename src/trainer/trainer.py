import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import random_split
from evaluation.metrics import compute_rouge, SummarizationMetrics
import wandb
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

class SummarizationModule(pl.LightningModule):
    def __init__(self, model, tokenizer, config):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.metrics = SummarizationMetrics(config)
        self.save_hyperparameters(config)
        self.validation_step_outputs = []  # 검증 출력 저장용
        
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.model._shared_step(batch, batch_idx)
        
        # batch_size 계산
        batch_size = len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
        
        # WandB에 학습 지표 로깅 (batch_size 명시)
        self.log('train_loss', loss, 
                 on_step=True, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True,
                 batch_size=batch_size)  # batch_size 추가
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, references, predictions = self.model._shared_step(batch, batch_idx)
        
        # batch_size 계산
        batch_size = len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
        
        # WandB에 검증 지표 로깅 (batch_size 명시)
        self.log('val_loss', loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True,
                 batch_size=batch_size)
        
        # 메트릭 계산 및 로깅
        if predictions is not None and references is not None:
            metrics = self.metrics.compute_metrics(predictions, references)
            for metric_name, value in metrics.items():
                self.log(f'val_{metric_name}', value, 
                        on_step=False, 
                        on_epoch=True, 
                        prog_bar=True, 
                        logger=True,
                        batch_size=batch_size)
        
        # 출력 저장
        self.validation_step_outputs.append({
            'val_loss': loss,
            'references': references,
            'predictions': predictions,
            'batch_size': batch_size
        })
        
        return {
            'val_loss': loss,
            'metrics': metrics if 'metrics' in locals() else {},
            'batch_size': batch_size
        }
    
    def on_validation_epoch_end(self):
        """검증 에폭 종료 시 전체 메트릭 계산"""
        if not self.validation_step_outputs:
            return
        
        # 전체 예측과 참조 수집
        all_predictions = []
        all_references = []
        total_val_loss = 0
        num_batches = 0
        
        for output in self.validation_step_outputs:
            if output['predictions'] is not None and output['references'] is not None:
                all_predictions.extend(output['predictions'])
                all_references.extend(output['references'])
            total_val_loss += output['val_loss']
            num_batches += 1
        
        # 평균 validation loss 계산
        avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
        
        # 샘플 출력 (첫 번째 배치의 예시)
        if all_predictions and all_references:
            print(f"\n=== Validation Epoch {self.current_epoch} ===")
            for i in range(min(3, len(all_predictions))):
                print(f"\nTarget: {all_references[i]}")
                print(f"Generated: {all_predictions[i]}")
                print("-" * 50)
            
            # 전체 메트릭 계산 및 출력
            metrics = self.metrics.compute_metrics(all_predictions, all_references)
            print(f"\nEpoch {self.current_epoch} Metrics:")
            print(f"Val Loss: {avg_val_loss:.4f}")
            for k, v in metrics.items():
                self.log(f"epoch_val_{k}", v, on_epoch=True, logger=True)
                print(f"{k}: {v:.4f}")
            print("-" * 50)
        
        # 메모리 정리
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        
        # 스케줄러 설정
        if hasattr(self.config.trainer, 'optimizer') and hasattr(self.config.trainer.optimizer, 'lr_scheduler'):
            scheduler_config = self.config.trainer.optimizer.lr_scheduler
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=scheduler_config.warmup_steps,
                num_training_steps=scheduler_config.total_steps
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        
        return optimizer

    def _format_prompt(self, dialogue, model_type):
        """모델별 최적화된 프롬프트 형식 반환"""
        if model_type == "LlamaSummarizer":
            return f"### Instruction: Summarize the following dialogue.\n\n### Input:\n{dialogue}\n\n### Response:"
        elif model_type == "T5Summarizer":
            return f"summarize: {dialogue}"
        else:  # BART 등 다른 모델
            return f"Summarize the following dialogue:\n{dialogue}"

    def _shared_step(self, batch, batch_idx=None):
        dialogues, summaries = batch
        
        # 1. 모델별 최적화된 프롬프트 구성
        model_type = self.model.__class__.__name__
        prompts = [
            self._format_prompt(dialogue, model_type)
            for dialogue in dialogues
        ]
        
        # 2. 입력 텍스트 토크나이징
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.preprocessing.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # 3. 레이블(타겟) 토크나이징
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                summaries,  # 요약 텍스트만 레이블로 사용
                padding=True,
                truncation=True,
                max_length=self.config.preprocessing.max_length,
                return_tensors="pt"
            ).input_ids.to(self.device)
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # forward pass 전에 train/eval 모드 설정
        if self.training:
            self.model.train()
            outputs = self.model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )
            loss = outputs.loss
            return loss, None, None
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels
                )
                loss = outputs.loss
                
                # 첫 번째 배치의 첫 번째 예시에 대해서만 생성
                if batch_idx == 0:
                    generated_summary = self.model.generate_summary(dialogues[0])
                    return loss, [generated_summary], [summaries[0]]
                
                return loss, None, None

    def on_validation_epoch_start(self):
        """검증 에폭 시작 시 출력 저장용 리스트 초기화"""
        self.validation_step_outputs = []

    def on_train_epoch_end(self):
        """에폭 종료 시 추가 메트릭 계산 및 시각화"""
        # 학습 곡선 시각화
        if self.current_epoch % self.config.trainer.get('plot_every_n_epochs', 1) == 0:
            fig = self.plot_training_curves()
            self.logger.experiment.log({
                "train/rouge_scores": wandb.Image(fig),  # matplotlib figure를 wandb.Image로 변환
                "epoch": self.current_epoch
            })
            plt.close(fig)  # 메모리 누수 방지를 위해 figure 닫기

    def plot_training_curves(self):
        """학습 곡선 시각화"""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import torch
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # 손실 곡선
            metrics = self.trainer.callback_metrics
            print(f"Available metrics: {list(metrics.keys())}")  # 디버깅용
            
            # 텐서를 리스트로 변환 (numpy 대신)
            def to_list(x):
                if torch.is_tensor(x):
                    try:
                        return x.detach().cpu().tolist()
                    except:
                        return float(x)  # 스칼라 텐서의 경우
                if isinstance(x, (list, tuple)):
                    return x
                return [float(x)]  # 단일 텐서의 경우
            
            # 손실값 가져기 (없으면 빈 리스트)
            train_loss = to_list(metrics.get('train_loss', 0.0))
            val_loss = to_list(metrics.get('val_loss', 0.0))
            
            # 리스트가 아닌 경우 리스트로 변환
            if not isinstance(train_loss, list):
                train_loss = [train_loss]
            if not isinstance(val_loss, list):
                val_loss = [val_loss]
            
            print(f"Train loss: {train_loss}")  # 디버깅용
            print(f"Val loss: {val_loss}")      # 디버깅용
            
            # 손실 곡선 그리기
            epochs = range(len(train_loss))
            ax1.plot(epochs, train_loss, label='Train')
            ax1.plot(epochs, val_loss, label='Validation')
            ax1.set_title('Loss Curves')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            
            # ROUGE 스코어
            if 'val_rouge1' in metrics:
                rouge1 = to_list(metrics['val_rouge1'])
                rouge2 = to_list(metrics['val_rouge2'])
                rougeL = to_list(metrics['val_rougeL'])
                
                if not isinstance(rouge1, list): rouge1 = [rouge1]
                if not isinstance(rouge2, list): rouge2 = [rouge2]
                if not isinstance(rougeL, list): rougeL = [rougeL]
                
                epochs = range(len(rouge1))
                ax2.plot(epochs, rouge1, label='ROUGE-1')
                ax2.plot(epochs, rouge2, label='ROUGE-2')
                ax2.plot(epochs, rougeL, label='ROUGE-L')
                ax2.set_title('ROUGE Scores')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Score')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'ROUGE scores not available yet', 
                        horizontalalignment='center', verticalalignment='center')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            import traceback
            print(f"Warning: Failed to plot training curves")
            print(f"Error: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            return None
