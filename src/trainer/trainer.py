import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from torch.utils.data import random_split
from evaluation.metrics import compute_rouge, SummarizationMetrics

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
        
        # 모델 파라미터 수를 계산하기 위해 model 속성 추가
        self.model = self.summarizer.model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.summarizer.model.parameters(),
            lr=self.config.model.learning_rate,
            weight_decay=self.config.model.weight_decay
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        dialogues, summaries = batch
        tokenized = self.tokenizer(
            list(dialogues), 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                list(summaries), 
                padding=True, 
                truncation=True, 
                max_length=128,
                return_tensors="pt"
            ).input_ids

        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        labels = labels.to(self.device)

        outputs = self.summarizer.model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            labels=labels
        )

        loss = outputs.loss
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(dialogues))
        return loss

    def validation_step(self, batch, batch_idx):
        dialogues, summaries = batch
        tokenized = self.tokenizer(
            list(dialogues), 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                list(summaries), 
                padding=True, 
                truncation=True, 
                max_length=128,
                return_tensors="pt"
            ).input_ids

        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        labels = labels.to(self.device)

        outputs = self.summarizer.model(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            labels=labels
        )

        loss = outputs.loss
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, batch_size=len(dialogues))
        
        batch_size = len(dialogues)
        
        generated_ids = self.summarizer.model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_length=self.config.model.max_length,
            num_beams=self.config.model.num_beams,
            early_stopping=True
        )
        
        generated_summaries = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        metric_scores = self.metrics.compute_metrics(
            predictions=generated_summaries,
            references=summaries
        )
        
        # 생성된 요약� 샘플 로깅 (매 N 스텝마다)
        if batch_idx % self.config.trainer.get('log_every_n_steps', 100) == 0:
            sample_idx = 0  # 첫 �째 배치 아이템
            self.logger.experiment.log_text(
                f"Original: {dialogues[sample_idx]}\n"
                f"Generated: {generated_summaries[sample_idx]}\n"
                f"Reference: {summaries[sample_idx]}"
            )
        
        # 메트릭 로깅
        for metric_name, score in metric_scores.items():
            self.log(
                f"val_{metric_name}",
                score,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True
            )
        
        return loss

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
        """학습 곡선 시각��"""
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
