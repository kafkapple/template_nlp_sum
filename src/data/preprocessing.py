import re
from typing import Optional, List

def clean_text(text: str, config: Optional[dict] = None) -> str:
    """향상된 텍스트 전처리"""
    if not config:
        config = {}
    
    # 기본 클리닝
    if config.get("basic_cleaning", True):
        # 여러 줄 공백을 하나로
        text = re.sub(r'\n+', '\n', text)
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        
    # 특수문자 처리
    if config.get("remove_special_chars", False):
        text = re.sub(r'[^a-zA-Z0-9가-힣\s\.,!?"\'-]', '', text)
    
    # 대화 마커 정규화
    if config.get("normalize_markers", True):
        text = re.sub(r'#\s*Person\s*\d+\s*:', 'Speaker:', text)
        text = re.sub(r'Speaker\s+\d+:', 'Speaker:', text)
    
    # 문장 부호 정규화
    if config.get("normalize_punctuation", True):
        text = re.sub(r'\.{2,}', '...', text)  # 말줄임표 정규화
        text = re.sub(r'[.!?]+([.!?])', r'\1', text)  # 중복 문장부호 제거
    
    return text.strip()

def truncate_dialogue(dialogue: str, max_words: int = 100) -> str:
    """대화 길이 제한"""
    words = dialogue.split()
    if len(words) > max_words:
        return ' '.join(words[:max_words]) + '...'
    return dialogue 