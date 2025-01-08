import wandb
from typing import Any, Dict, Optional
import pytorch_lightning as pl
from .base_logger import BaseLogger

class WandBLogger(pl.LightningLoggerBase, BaseLogger):
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        save_dir: str = "logs",
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__()
        self.project = project
        self.name = name
        self.save_dir = save_dir
        self.config = config
        
        # WandB 초기화
        wandb.init(
            project=project,
            name=name,
            dir=save_dir,
            config=config,
            **kwargs
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """메트릭 로깅"""
        wandb.log(metrics, step=step)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        wandb.config.update(params)
    
    def log_artifact(self, artifact_name: str, artifact_path: str, artifact_type: str):
        """모델 체크포인트 등 아티팩트 로깅"""
        artifact = wandb.Artifact(
            artifact_name,
            type=artifact_type
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
    
    def log_text(self, text: str, step: Optional[int] = None):
        """텍스트 로깅"""
        wandb.log({"text": wandb.Html(text)}, step=step)
    
    def log_figure(self, figure_name: str, figure, step: Optional[int] = None):
        """시각화 결과 로깅"""
        wandb.log({figure_name: figure}, step=step)

    @property
    def experiment(self):
        """Return the experiment object associated with this logger."""
        return wandb.run

    @property
    def version(self):
        """Return the experiment version."""
        return wandb.run.id if wandb.run else None

    @property
    def name(self):
        """Return the experiment name."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the experiment name."""
        self._name = value
