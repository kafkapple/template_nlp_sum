from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pytorch_lightning as pl

class BaseLogger(ABC):
    """기본 로거 클래스"""
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """메트릭 로깅"""
        pass
    
    @abstractmethod
    def log_hyperparams(self, params: Dict[str, Any]):
        """하이퍼파라미터 로깅"""
        pass
    
    @abstractmethod
    def log_artifact(self, artifact_name: str, artifact_path: str, artifact_type: str):
        """모델 체크포인트 등 아티팩트 로깅"""
        pass
    
    @abstractmethod
    def log_text(self, text: str, step: Optional[int] = None):
        """텍스트 로깅 (예: 생성된 요약문)"""
        pass
    
    @abstractmethod
    def log_figure(self, figure_name: str, figure, step: Optional[int] = None):
        """시각화 결과 로깅"""
        pass
