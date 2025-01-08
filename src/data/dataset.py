import json
import re
import os
from torch.utils.data import Dataset
from .download import ensure_dialogsum_files

def basic_cleaning(text, remove_special=True, lower_case=False):
    text = re.sub(r"[^a-zA-Z0-9가-힣\s\.,\?]", "", text) if remove_special else text
    text = text.lower() if lower_case else text
    text = re.sub(r"\s+", " ", text).strip()
    return text

class DialogueSumDataset(Dataset):
    def __init__(self, file_name, tokenizer, max_length=512, preprocessing_cfg=None, dataset_dir=None):
        """
        file_name: 예) 'dialogsum.train.jsonl'
        dataset_dir: 데이터셋이 저장될 폴더 경로
        """
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocessing_cfg = preprocessing_cfg if preprocessing_cfg else {}

        # 1) DialogSum 파일 자동 다운로드
        if dataset_dir is not None:
            ensure_dialogsum_files(dataset_dir)  
            file_path = os.path.join(dataset_dir, file_name)
        else:
            file_path = file_name  # dataset_dir 없으면 직접 file_name만 사용
        
        # 2) 파일 열기
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    dialogue = data.get("dialogue", "")
                    summary = data.get("summary", "")
                    # 전처리
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
                    self.samples.append((dialogue, summary))
        except FileNotFoundError:
            print(f"[Warning] File not found: {file_path}")
        except json.JSONDecodeError:
            print(f"[Warning] JSON decoding error in file: {file_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
