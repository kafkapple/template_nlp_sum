import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import random_split
from evaluation.metrics import compute_rouge, SummarizationMetrics
import wandb

class SummarizationModule(pl.LightningModule):
    def __init__(self, model, tokenizer, config):
        super().__init__()
        self.summarizer = model
        self.tokenizer = tokenizer
        self.config = config
        self.metrics = SummarizationMetrics(config)
        
        # config를 yaml 형태로 변환하여 저장
        config_dict = OmegaConf.to_container(config, resolve=True)
        self.save_hyperparameters({
            "config": config_dict,
            "model_type": self.summarizer.__class__.__name__
        })

    def configure_optimizers(self):
        # 학습 가능한 파라미터만 optimizer에 전달
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.summarizer.model.parameters()),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        return optimizer

    def _shared_step(self, batch, batch_idx=None):
        dialogues, summaries = batch
        
        # 프롬프트 형식으로 입력 구성
        prompts = [
            f"Summarize the following dialogue:\n{dialogue}\nSummary: {summary}"
            for dialogue, summary in zip(dialogues, summaries)
        ]
        
        # 토크나이징
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.preprocessing.max_length,
            return_tensors="pt"
        ).to(self.device)
        
        # 레이블 생성 (입력의 -1을 제외한 모든 토큰)
        labels = inputs["input_ids"].clone()
        # -100은 loss 계산에서 무시됨
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # forward pass 전에 train/eval 모드 설정
        if self.training:
            self.summarizer.train()
            outputs = self.summarizer.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=labels
            )
            loss = outputs.loss
            return loss, None, None
        else:
            self.summarizer.eval()
            with torch.no_grad():
                outputs = self.summarizer.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=labels
                )
                loss = outputs.loss
                
                # 첫 번째 배치의 첫 번째 예시에 대해서만 생성
                if batch_idx == 0:
                    generated_summary = self.summarizer.generate_summary(dialogues[0])
                    return loss, [generated_summary], [summaries[0]]
                
                return loss, None, None

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(batch[0]))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, generated_texts, target_texts = self._shared_step(batch, batch_idx)
        
        # 샘플 텍스트 로깅 (첫 번째 배치의 첫 번째 예시만)
        if batch_idx == 0 and generated_texts is not None:
            wandb.log({
                "example_generation": wandb.Table(
                    columns=["Generated", "Target"],
                    data=[[generated_texts[0], target_texts[0]]]
                )
            })
        
        self.log("val_loss", loss, prog_bar=True, batch_size=len(batch[0]))
        return {"val_loss": loss, "generated_texts": generated_texts, "target_texts": target_texts}

    def on_train_epoch_end(self):
        """에폭 종료 시 추가 메트릭 계산 및 시각화"""
        # 학습 곡선 시각화
        if self.current_epoch % self.config.trainer.get('plot_every_n_epochs', 1) == 0:
            fig = self.plot_training_curves()
            self.logger.experiment.log_figure(
                "training_curves",
                fig,
                self.current_epoch
            )

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
