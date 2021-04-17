import argparse

import pandas as pd

import torch
import torch.nn as nn

from utils.models import EfficientNetModel
from utils.transforms import get_valid_transform
from utils.loaders import get_valid_loader

from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

from tqdm import tqdm

batch_size = 32

df = pd.read_csv("labeled_part.csv")
df["label"] = df["label"].apply(lambda x: 1 if x == "logo" else 0)

valid_transform = get_valid_transform(img_size=224)
valid_loader = get_valid_loader(
    df,
    "label",
    valid_transform,
    batch_size=batch_size,
    num_workers=0,
    shuffle=False
)

model = EfficientNetModel(loss=nn.CrossEntropyLoss(), num_classes=2)
model.load_from_checkpoint("epoch=2_val_loss=0.6148.ckpt", loss=nn.CrossEntropyLoss())
model.eval()

softmax = nn.Softmax(dim=-1)
pred = []
with tqdm(ascii=True, leave=False, total=len(valid_loader)) as bar:
    for i, batch in enumerate(valid_loader):

        images, y = batch["x"], batch["y"]
        with torch.no_grad():

            y_hat = model(images)
            y_hat = softmax(y_hat)
        
        pred += y_hat[:, 1].tolist()
        
        bar.update()

df["pred"] = pred
df.to_csv("labeled_predicted.csv", index=False)

precision = precision_score(df.label, df.pred > 0.5)
recall = recall_score(df.label, df.pred > 0.5)
f1 = f1_score(df.label, df.pred > 0.5)
conf_matrix = confusion_matrix(df.label, df.pred > 0.5)
print(f"Precision = {precision:.4f}")
print(f"Recall = {recall:.4f}")
print(f"F1 = {f1:.4f}")
print("Confusion matrix:")
print(conf_matrix)