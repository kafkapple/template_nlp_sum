import wandb
import pytorch_lightning as pl
from typing import Dict, Any, Optional
from omegaconf import OmegaConf

class WandBLogger(pl.loggers.WandbLogger):
    def __init__(
        self,
        project: str,
        name: str,
        save_dir: str,
        config: Dict[str, Any]
    ):
        # Hydra config를 dict로 변환
        if hasattr(config, "_content"):  # OmegaConf 객체인 경우
            wandb_config = OmegaConf.to_container(
                config,
                resolve=True,
                enum_to_str=True
            )
        else:
            wandb_config = config
        
        # WandB 초기화
        super().__init__(
            project=project,
            name=name,
            save_dir=save_dir
        )
        
        # config 업데이트
        if wandb.run is not None:
            wandb.config.update(wandb_config, allow_val_change=True)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """메트릭 로깅"""
        super().log_metrics(metrics, step=step)
    
    def log_hyperparams(self, params: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        # OmegaConf 객체를 dict로 변환
        if hasattr(params, "_content"):
            params = OmegaConf.to_container(
                params,
                resolve=True,
                enum_to_str=True
            )
        
        if wandb.run is not None:
            wandb.config.update(params, allow_val_change=True)
    
    def finalize(self, status: str):
        """실험 종료"""
        if wandb.run is not None:
            wandb.finish(exit_code=0 if status == "success" else 1)
