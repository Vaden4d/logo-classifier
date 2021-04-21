import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """PyTorch dataset - retieve images using .csv file"""
    def __init__(self, csv, transform, target_column):
        super().__init__()
        self.csv = csv
        self.target_column = target_column
        self.size = len(csv)
        self.transform = transform
        if self.target_column is not None:
            self.num_classes = len(self.csv.groupby(self.target_column))

    def __len__(self):
        return self.size
  
    def __getitem__(self, idx):
        row = self.csv.iloc[int(idx)]
        
        img = np.array(Image.open(row.path).convert("RGB"))
        img = self.transform(image=img)
        image = img["image"]
        
        if self.target_column is None:
            return {
                "features": image
            }
        else:
            label = row[self.target_column]
            return {
                "features": image,
                "targets": F.one_hot(
                    torch.Tensor([label]).long(), 
                    num_classes=self.num_classes
                )[0],
            }