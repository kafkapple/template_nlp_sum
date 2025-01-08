from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from torch import nn
from .base_model import BaseModel

class BartSummarizer(BaseModel):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        finetune_cfg = config.finetune_strategy
        
        # 모델 로드
        self.model = BartForConditionalGeneration.from_pretrained(
            model_cfg.name,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if model_cfg.get("use_fp16", False) else torch.float32,
        )
        
        # 먼저 모든 ��이어 이름 출력
        print("\nAvailable layers:")
        for name, _ in self.model.named_parameters():
            print(f"  - {name}")
        
        # 1. 먼저 모� 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. 특정 레이어만 학습 가능하도록 설정
        trainable_layers = getattr(finetune_cfg, "unfreeze_layers", [
            "model.decoder.layers.11.self_attn.q_proj",
            "model.decoder.layers.11.self_attn.v_proj",
            "model.decoder.layers.11.fc1",
            "model.decoder.layers.11.fc2",
            "model.decoder.layers.10.self_attn.q_proj",
            "model.decoder.layers.10.self_attn.v_proj",
            "model.decoder.layers.10.fc1",
            "model.decoder.layers.10.fc2",
            "model.shared",
            "final_logits_bias"
        ])
        
        # 3. 학습 가능한 레이어 설정 및 확인
        trainable_found = False
        for name, param in self.model.named_parameters():
            # 정확한 이름 매칭을 위해 'model.' 접두사 추가
            full_name = f"model.{name}" if not name.startswith("model.") else name
            if any(layer in full_name for layer in trainable_layers):
                param.requires_grad = True
                trainable_found = True
                print(f"Trainable layer found: {name}")
        
        if not trainable_found:
            print("\nWarning: No trainable layers found! Available layers:")
            for name, _ in self.model.named_parameters():
                print(f"  - model.{name}")
        
        # 4. 모델을 �시적으로 train 모드로 설정
        self.model.train()
        
        # 5. gradient checkpointing 활성�
        if getattr(finetune_cfg, "gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
        
        # 6. 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        self.tokenizer = BartTokenizer.from_pretrained(model_cfg.tokenizer_name)
        self.max_length = model_cfg.max_length
        self.num_beams = model_cfg.num_beams

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate_summary(self, text):
        # 입력 텍스트 토크나이징
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            max_length=self.max_length, 
            truncation=True
        )
        
        # 모든 텐서를 모델과 같은 디바이스로 이동
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 요약 생성
        summary_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True
        )
        
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
