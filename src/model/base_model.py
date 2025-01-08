from abc import ABC
import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import torch.nn as nn

class BaseModel(ABC):
    def setup_quantization(self, finetune_cfg):
        """양자화 설정"""
        quant_mode = finetune_cfg.get("quantization", "none")
        
        if quant_mode == "qlora":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quant_mode == "8bit":
            return BitsAndBytesConfig(load_in_8bit=True)
        return None

    def apply_lora(self, model, finetune_cfg, model_type, task_type):
        """LoRA 적용 (QLoRA인 경우에만)"""
        if finetune_cfg.get("quantization") != "qlora":
            return model
            
        model = prepare_model_for_kbit_training(model)
        lora_cfg = finetune_cfg.get("lora_config", {})
        
        target_modules = lora_cfg.get("target_modules", {}).get(model_type, ["q_proj", "v_proj"])
        
        config = LoraConfig(
            r=lora_cfg.get("rank", 8),
            lora_alpha=lora_cfg.get("alpha", 32),
            target_modules=target_modules,
            lora_dropout=lora_cfg.get("dropout", 0.1),
            bias="none",
            task_type=task_type
        )
        
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
        return model 

    def setup_layer_freezing(self, model, fine_tuning_cfg):
        """레이어 동결/해제 설정"""
        # 1. 먼저 모든 파라미터 동결
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. 임베딩 레이어 동결 설정
        if not fine_tuning_cfg.freeze_embeddings:
            for name, param in model.named_parameters():
                if "embed" in name:
                    param.requires_grad = True
                    
        # 3. 지정된 레이어 해동
        unfreeze_layers = fine_tuning_cfg.unfreeze_layers
        for name, param in model.named_parameters():
            for layer in unfreeze_layers:
                if layer in name:
                    param.requires_grad = True
                    break
        
        # 4. Gradient Checkpointing 설정
        if fine_tuning_cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            
        return model

    def print_trainable_parameters(self):
        """학습 가능한 파라미터 정보 출력"""
        trainable_params = 0
        all_params = 0
        for name, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable layer: {name}")
        
        print(
            f'trainable params: {trainable_params:,d} || '
            f'all params: {all_params:,d} || '
            f'trainable%: {100 * trainable_params / all_params:.2f}%'
        ) 