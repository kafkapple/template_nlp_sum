from .bart import BartSummarizer
from .t5 import T5Summarizer
from .llama import LlamaSummarizer

def get_model(model_name: str, config):
    if "bart" in model_name.lower():
        return BartSummarizer(config)
    elif "t5" in model_name.lower():
        return T5Summarizer(config)
    elif "llama" in model_name.lower():
        return LlamaSummarizer(config)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
