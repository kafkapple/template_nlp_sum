from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=16, shuffle=True):
    """길이 기반 배치 구성"""
    
    def collate_fn(batch):
        # 길이 기준으로 정렬
        batch.sort(key=lambda x: len(x[0].split()), reverse=True)
        dialogues, summaries = zip(*batch)
        
        return list(dialogues), list(summaries)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
