from torch.utils.data import DataLoader

def create_dataloader(dataset, batch_size=4, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
