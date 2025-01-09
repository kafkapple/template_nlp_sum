# src/main.py
import os
# OpenMP ��류 방지를 위한 환경 변수 설정
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from data.dataset import DialogueSumDataset
from data.dataloader import create_dataloader
from model.model_factory import get_model
from trainer.trainer import SummarizationModule
import warnings
from omegaconf import DictConfig, OmegaConf
warnings.filterwarnings("ignore")

@hydra.main(version_base=None, config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig):
    # 1. data config 불러오기 (기본 defaults에서 dialogsum로 설정되어 있음)
    data_cfg = cfg.dataset
    model_cfg = cfg.model
    train_cfg = cfg.trainer

    # 2. 모델 준비
    model = get_model(model_cfg.name, cfg)  # or pass entire cfg

    # 3. Dataset 로드 (자동 다운로드 포함)
    train_dataset = DialogueSumDataset(
        file_name=data_cfg.train_file,
        tokenizer=model.tokenizer,
        max_length=data_cfg.max_length,
        preprocessing_cfg=cfg.preprocessing,   # 전처리 설정
        dataset_dir=data_cfg.dataset_dir       # 자동 다운로드 폴더
    )
    val_dataset = DialogueSumDataset(
        file_name=data_cfg.dev_file,
        tokenizer=model.tokenizer,
        max_length=data_cfg.max_length,
        preprocessing_cfg=cfg.preprocessing,
        dataset_dir=data_cfg.dataset_dir
    )

    # 4. DataLoader
    train_loader = create_dataloader(train_dataset, batch_size=data_cfg.batch_size)
    val_loader = create_dataloader(val_dataset, batch_size=data_cfg.batch_size, shuffle=False)

    # 5. Lightning 모듈
    lightning_model = SummarizationModule(
        model=model,
        tokenizer=model.tokenizer,
        config=cfg
    )

    # 6. Logger 설정
    from logger.wandb_logger import WandBLogger
    wandb_logger = WandBLogger(
        project=train_cfg.logging.wandb.project,
        name=train_cfg.logging.wandb.name,
        save_dir=train_cfg.logging.save_dir,
        config=cfg
    )

    # 7. Checkpoint 콜백 설정
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        **OmegaConf.to_container(train_cfg.checkpoint, resolve=True)
    )

    # 8. Trainer 설정 #I love you
    trainer = pl.Trainer(
        max_epochs=train_cfg.max_epochs,
        gpus=train_cfg.gpus,
        precision=train_cfg.precision,
        accumulate_grad_batches=train_cfg.accumulate_grad_batches,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        default_root_dir=train_cfg.logging.save_dir
    )

    # 9. 학습
    trainer.fit(lightning_model, train_loader, val_loader)

if __name__ == "__main__":
    main()
