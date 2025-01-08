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
        
        # 1. 데이터를 ��바른 디바이스로 이동
        tokenized = self.tokenizer(
            list(dialogues), 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                list(summaries), 
                padding=True, 
                truncation=True, 
                max_length=128,
                return_tensors="pt"
            ).input_ids.to(self.device)
        
        # 2. forward pass 전에 train/eval ��드 설정
        if self.training:
            self.summarizer.train()
            # training 모드에서는 gradient 계산이 필요
            outputs = self.summarizer.model(
                input_ids=tokenized["input_ids"],
                attention_mask=tokenized["attention_mask"],
                labels=labels
            )
            loss = outputs.loss
            assert loss.requires_grad, "Loss does not require gradients in training mode!"
            return loss, None, None
        else:
            # validation 모드
            self.summarizer.eval()
            with torch.no_grad():  # validation에서는 gradient 계산 불필요
                outputs = self.summarizer.model(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"],
                    labels=labels
                )
                loss = outputs.loss
                
                # 첫 번째 배��의 첫 번째 예시에 대해서만 생성
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
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 손실 곡선
        metrics = self.trainer.callback_metrics
        ax1.plot(metrics['train_loss'], label='Train')
        ax1.plot(metrics['val_loss'], label='Validation')
        ax1.set_title('Loss Curves')
        ax1.legend()
        
        # ROUGE 스코어
        ax2.plot(metrics['val_rouge1'], label='ROUGE-1')
        ax2.plot(metrics['val_rouge2'], label='ROUGE-2')
        ax2.plot(metrics['val_rougeL'], label='ROUGE-L')
        ax2.set_title('ROUGE Scores')
        ax2.legend()
        
        return fig
