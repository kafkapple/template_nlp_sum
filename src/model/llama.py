from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from .base_model import BaseModel

class LlamaSummarizer(BaseModel):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        finetune_cfg = config.finetune_strategy
        
        # 모델과 토크나이저 로드
        self.model = LlamaForCausalLM.from_pretrained(
            model_cfg.name,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(model_cfg.tokenizer_name)
        
        # 먼저 모든 레이어 이름 출력
        print("\nAvailable layers:")
        for name, _ in self.model.named_parameters():
            print(f"  - {name}")
        
        # 1. 먼저 모든 파라미터 동결
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 2. 특정 레이어만 학습 가능하도록 설정
        trainable_layers = getattr(finetune_cfg, "unfreeze_layers", [
            "model.layers.0",
            "model.layers.1",
            "lm_head"  # 출력 레이어는 항상 학습되도록
        ])
        
        trainable_found = False
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in trainable_layers):
                param.requires_grad = True
                trainable_found = True
                print(f"Trainable layer found: {name}")
        
        if not trainable_found:
            # 아무 레이어도 찾지 못했다면 마지막 레이어라도 학습하도록 설정
            print("\nNo trainable layers found. Enabling last layer for training...")
            for name, param in self.model.named_parameters():
                if "lm_head" in name or "layers.31" in name:  # 마지막 레이어
                    param.requires_grad = True
                    print(f"Enabled training for: {name}")
        
        # 3. gradient checkpointing 활성화
        if getattr(finetune_cfg, "gradient_checkpointing", True):
            self.model.gradient_checkpointing_enable()
        
        # 4. 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        self.max_length = model_cfg.max_length
        self.num_beams = model_cfg.num_beams

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate_summary(self, text):
        # 프롬프트 추가
        prompt = f"Summarize the following dialogue:\n{text}\nSummary:"
        
        # 입력 텍스트 토크나이징
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # 모든 텐서를 모델과 같은 디바이스로 이동
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 요약 생성
        outputs = self.model.generate(
            **inputs,
            max_length=self.max_length,
            num_beams=self.num_beams,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
