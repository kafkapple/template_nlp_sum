from torch.utils.data import DataLoader

def collate_fn(batch):
    # 배치 데이터 처리 로직
    dialogues, summaries = zip(*batch)
    return list(dialogues), list(summaries)

def create_dataloader(dataset, batch_size, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
