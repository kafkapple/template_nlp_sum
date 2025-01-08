from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from torch import nn
from .base_model import BaseModel

class BartSummarizer(BaseModel):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        finetune_cfg = config.finetune_strategy
        
        self.tokenizer = BartTokenizer.from_pretrained(model_cfg.tokenizer_name)
        
        # 양자화 설정
        quant_config = self.setup_quantization(finetune_cfg)
        
        # 모델 로드
        self.model = BartForConditionalGeneration.from_pretrained(
            model_cfg.name,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if model_cfg.get("use_fp16", False) else torch.float32
        )
        
        # QLoRA 설정
        if finetune_cfg.get("quantization") == "qlora":
            from peft import prepare_model_for_kbit_training
            
            # QLoRA를 위한 모� 준비
            self.model = prepare_model_for_kbit_training(
                self.model, 
                use_gradient_checkpointing=True
            )
            
            # LoRA 적용
            self.model = self.apply_lora(self.model, finetune_cfg, "bart", "SEQ_2_SEQ_LM")
        else:
            # 일반 학습의 경우 gradient checkpointing 활성화
            self.model.gradient_checkpointing_enable()
            
            # 모든 파라��터 학습 가능하도록 설정
            for param in self.model.parameters():
                if param.dtype in [torch.float16, torch.float32, torch.float64]:
                    param.requires_grad = True
        
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
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, truncation=True
        )
        summary_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
