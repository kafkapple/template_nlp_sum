import json
import re
import os
from torch.utils.data import Dataset
from .download import ensure_dialogsum_files
from datasets import load_dataset
from .preprocessing import clean_text

def basic_cleaning(text, remove_special=True, lower_case=False):
    text = re.sub(r"[^a-zA-Z0-9가-힣\s\.,\?]", "", text) if remove_special else text
    text = text.lower() if lower_case else text
    text = re.sub(r"\s+", " ", text).strip()
    return text

class DialogueSumDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length, preprocessing_cfg, dataset_dir):
        # DialogSum 데이터셋 자동 다운로드 및 로드
        self.dataset = load_dataset(
            "knkarthick/dialogsum",
            split=file_name,
            cache_dir=dataset_dir
        )
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing_cfg = preprocessing_cfg
        
        # 데이터 필터링
        if preprocessing_cfg.get("filter_data", False):
            self.dataset = self.dataset.filter(self._filter_sample)
        
        # 데이터 증강
        if preprocessing_cfg.get("augment_data", False):
            self.dataset = self._augment_dataset()
    
    def __len__(self):
        return len(self.dataset)
    
    def _filter_sample(self, sample):
        """데이터 품질 기반 필터링"""
        dialogue = sample['dialogue']
        summary = sample['summary']
        
        # 최소/최대 길이 확인
        if len(dialogue.split()) < 10 or len(dialogue.split()) > 1000:
            return False
        if len(summary.split()) < 5 or len(summary.split()) > 100:
            return False
            
        # 요약문이 대화문보다 길면 제외
        if len(summary.split()) > len(dialogue.split()) * 0.5:
            return False
            
        return True
    
    def _augment_dataset(self):
        """데이터 증강 기법 적용"""
        augmented_data = []
        
        for sample in self.dataset:
            # 원본 데이터 유지
            augmented_data.append(sample)
            
            # 화자 이름 변경
            if self.preprocessing_cfg.get("speaker_augmentation", False):
                dialogue = re.sub(r'Speaker:', 'Person:', sample['dialogue'])
                augmented_data.append({
                    'dialogue': dialogue,
                    'summary': sample['summary']
                })
        
        return augmented_data

    def __getitem__(self, idx):
        item = self.dataset[idx]
        dialogue = item['dialogue']
        summary = item['summary']
        
        # 전처리 적용
        if self.preprocessing_cfg.get("clean_text", True):
            dialogue = basic_cleaning(
                dialogue,
                remove_special=self.preprocessing_cfg.get("remove_special_chars", True),
                lower_case=self.preprocessing_cfg.get("lower_case", False)
            )
            summary = basic_cleaning(
                summary,
                remove_special=self.preprocessing_cfg.get("remove_special_chars", True),
                lower_case=self.preprocessing_cfg.get("lower_case", False)
            )
        
        return dialogue, summary
