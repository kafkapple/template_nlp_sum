from transformers import BartTokenizer, BartModel
import torch
from torch import nn
from .base_model import BaseModel

class CustomBartSummarizer(BaseModel):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.model
        
        # BART 토크나이저 로드
        self.tokenizer = BartTokenizer.from_pretrained(model_cfg.tokenizer_name)
        
        # BART 기본 모델 로드 (인코더-디코더 구조)
        self.base_model = BartModel.from_pretrained(model_cfg.name)
        
        # 기본 모델 전 프리즈
        self.base_model.eval()  # 평가 모드로 설정
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # 임베딩 차원
        hidden_size = self.base_model.config.hidden_size
        
        # 커스텀 디코더 레이어 추가 (이 부분만 학습됨)
        self.custom_decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                activation='gelu'
            ) for _ in range(3)
        ])
        
        # 출력 레이어 (이 부분도 학습됨)
        self.output_layer = nn.Linear(
            hidden_size, 
            len(self.tokenizer)
        )
        
        # 학습 가능한 파라미터 수 출력
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
        
        self.max_length = model_cfg.max_length
        self.num_beams = model_cfg.num_beams

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():  # base_model은 gradient 계산하지 않음
            # BART 인코더의 ��력 얻기
            encoder_outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            if labels is not None:
                # 디코더 입력 준비
                decoder_input_ids = labels[:, :-1]  # 마지막 토큰 제외
                decoder_attention_mask = (decoder_input_ids != self.tokenizer.pad_token_id)
                
                # BART 디코더의 초기 출력 얻기
                decoder_outputs = self.base_model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=attention_mask,
                    attention_mask=decoder_attention_mask,
                    return_dict=True
                )
                hidden_states = decoder_outputs.last_hidden_state

        # 여기서부터는 gradient �산
        # 커스텀 디코더 레이어 통과
        for decoder_layer in self.custom_decoder:
            hidden_states = decoder_layer(
                hidden_states,
                encoder_hidden_states,
                tgt_mask=self.generate_square_subsequent_mask(hidden_states.size(1)).to(hidden_states.device)
            )
        
        # 출력 레이어
        logits = self.output_layer(hidden_states)
        
        if labels is not None:
            # 손실 계산
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
            target_labels = labels[:, 1:].contiguous()  # 첫 번째 토큰(<s>) 제외
            loss = loss_fct(logits.view(-1, logits.size(-1)), target_labels.view(-1))
            
            return {
                "loss": loss,
                "logits": logits,
                "hidden_states": hidden_states,
                "encoder_last_hidden_state": encoder_hidden_states
            }
        
        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "encoder_last_hidden_state": encoder_hidden_states
        }

    def generate_summary(self, text):
        # 생성 로직 구현
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_length, truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            # 인코더 출��
            encoder_outputs = self.base_model.encoder(
                **inputs,
                return_dict=True
            )
            encoder_hidden_states = encoder_outputs.last_hidden_state
            
            # 디코더 �력 �잴�기화 (시작 토큰)
            decoder_input_ids = torch.tensor([[self.tokenizer.bos_token_id]]).to(self.device)
            
            # 자동 회귀적 생성
            for _ in range(self.max_length):
                decoder_outputs = self.base_model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=inputs["attention_mask"],
                    return_dict=True
                )
                hidden_states = decoder_outputs.last_hidden_state
                
                # 커스텀 디코더 ��과
                for decoder_layer in self.custom_decoder:
                    hidden_states = decoder_layer(
                        hidden_states,
                        encoder_hidden_states
                    )
                
                # 다음 토큰 �측
                logits = self.output_layer(hidden_states[:, -1:])
                next_token = logits.argmax(dim=-1)
                
                # 종료 토큰이면 중단
                if next_token[0, 0] == self.tokenizer.eos_token_id:
                    break
                    
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
        
        return self.tokenizer.decode(decoder_input_ids[0], skip_special_tokens=True)

    def generate_square_subsequent_mask(self, sz):
        """디코더의 self-attention을 위한 마스크 생성"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask 