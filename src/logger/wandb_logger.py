import wandb
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from omegaconf import OmegaConf
from .base_logger import BaseLogger

class WandBLogger(pl.loggers.WandbLogger, BaseLogger):
    def __init__(
        self,
        project: str,
        name: str,
        save_dir: str,
        config: Dict[str, Any]
    ):
        # Hydra config를 일반 dict로 변환
        wandb_config = OmegaConf.to_container(config, resolve=True)
        
        wandb.init(
            project=project,
            name=name,
            dir=save_dir,
            config=wandb_config
        )
        
        super().__init__(
            project=project,
            name=name,
            save_dir=save_dir,
            config=wandb_config
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """메트릭 로깅"""
        if wandb.run is None:
            wandb.init()
        wandb.log(metrics, step=step)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        if wandb.run is None:
            wandb.init()
        wandb.config.update(params)
    
    def log_artifact(self, artifact_name: str, artifact_path: str, artifact_type: str):
        """모델 체크포인트 등 아티팩트 로깅"""
        if wandb.run is None:
            wandb.init()
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type
        )
        artifact.add_file(artifact_path)
        wandb.log_artifact(artifact)
    
    def log_text(self, text: str, step: Optional[int] = None):
        """텍스트 로깅"""
        if wandb.run is None:
            wandb.init()
        # wandb.Table을 사용하여 텍스트를 로깅
        table = wandb.Table(columns=["Generated Text"])
        table.add_data(text)
        wandb.log({"generated_text": table}, step=step)
    
    def log_figure(self, figure_name: str, figure, step: Optional[int] = None):
        """시각화 결과 로깅"""
        if wandb.run is None:
            wandb.init()
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
