import torch
import psutil
import os

def get_gpu_memory_usage():
    """GPU 메모리 사용량 확인"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # MB 단위

def get_ram_usage():
    """RAM 사용량 확인"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # MB 단위

def print_memory_usage():
    """메모리 사용량 출력"""
    gpu_mem = get_gpu_memory_usage()
    ram_mem = get_ram_usage()
    print(f"GPU Memory: {gpu_mem:.2f}MB")
    print(f"RAM Usage: {ram_mem:.2f}MB") 


# 리소스 사용량, 연산 시간 도 메트릭으로 추가. 
