from torch.utils.data import DataLoader
from .dataset import ImageDataset

def get_loader(csv,
                target_column,
                transform,
                batch_size,
                num_workers,
                shuffle):
    dataset = ImageDataset(csv,
                            target_column=target_column,
                            transform=transform)
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle)
    return loader          


def get_loaders(train, 
                test,
                train_transform,
                test_transform,
                target_column,
                batch_size,
                num_workers,
                shuffle
                ):

    train_dataset = ImageDataset(train, 
                                target_column=target_column,
                                transform=train_transform)
    valid_dataset = ImageDataset(test,
                                target_column=target_column,
                                transform=test_transform)

    train_loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=shuffle)

    valid_loader = DataLoader(valid_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False)

    return train_loader, valid_loader

