import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, csv, transform, label_column="weak_label"):
        super().__init__()
        self.csv = csv
        self.label_column = label_column
        self.size = len(csv)
        self.transform = transform

    def __len__(self):
        return self.size
  
    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        path, label = row.path, row[self.label_column]
        img = np.array(Image.open(path).convert("RGB"))
        img = self.transform(image=img)
        image = img["image"]
        return {
            "x": image,
            "y": label,
        }