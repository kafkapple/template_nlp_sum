from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from torch import nn
from .base_model import BaseModel

class T5Summarizer(BaseModel):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        finetune_cfg = config.finetune_strategy
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_cfg.tokenizer_name)
        
        # 양자화 설정
        quant_config = self.setup_quantization(finetune_cfg)
        
        # 모델 로드
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_cfg.name,
            quantization_config=quant_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.float16 if model_cfg.get("use_fp16", False) else torch.float32
        )
        
        # LoRA 적용 (QLoRA인 경우)
        self.model = self.apply_lora(self.model, finetune_cfg, "t5", "SEQ_2_SEQ_LM")
        
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
        # T5는 요약 시 "summarize: " prefix 사용 권장
        prepended_text = "summarize: " + text
        inputs = self.tokenizer(
            prepended_text, return_tensors="pt", max_length=self.max_length, truncation=True
        )
        summary_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
