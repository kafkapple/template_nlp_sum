import os
import requests

DIALOGSUM_FILES = {
    "dialogsum.train.jsonl": "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.train.jsonl",
    "dialogsum.dev.jsonl": "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.dev.jsonl",
    "dialogsum.test.jsonl": "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.test.jsonl",

    # 필요 시 추가:
    # "dialogsum.hiddentest.dialogue.jsonl": "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.hiddentest.dialogue.jsonl",
    # "dialogsum.hiddentest.topic.jsonl":    "https://raw.githubusercontent.com/cylnlp/dialogsum/main/DialogSum_Data/dialogsum.hiddentest.topic.jsonl"
}

def download_file(url, save_path):
    """
    url로부터 파일을 다운로드해 save_path에 저장.
    requests 라이브러리 사용, 단순 예시.
    """
    print(f"[Info] Downloading from {url} to {save_path} ...")
    resp = requests.get(url, stream=True)
    resp.raise_for_status()  # 실패 시 예외 발생
    with open(save_path, 'wb') as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    print("[Info] Download complete!")

def ensure_dialogsum_files(dataset_dir):
    """
    dataset_dir 경로 내에 필요한 DialogSum 파일이 없으면 자동 다운로드.
    DIALOGSUM_FILES 딕셔너리 참조.
    """
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    for file_name, file_url in DIALOGSUM_FILES.items():
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(file_path):
            download_file(file_url, file_path)
        else:
            print(f"[Info] {file_name} already exists. Skipping download.")
