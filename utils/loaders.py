from torch.utils.data import DataLoader
from .dataset import ImageDataset

def get_valid_loader(test,
                    label_column,
                    test_transform,
                    batch_size,
                    num_workers,
                    shuffle):
    valid_dataset = ImageDataset(test,
                                label_column=label_column,
                                transform=test_transform)
    valid_loader = DataLoader(valid_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        shuffle=False)
    return valid_loader              


def get_loaders(train, 
                test,
                train_transform,
                test_transform,
                batch_size,
                num_workers,
                shuffle
                ):

    train_dataset = ImageDataset(train, 
                                transform=train_transform)
    valid_dataset = ImageDataset(test,
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