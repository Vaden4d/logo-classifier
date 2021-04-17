import os
import torch
import pandas as pd
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 

from utils.models import EfficientNetModel
from utils.transforms import get_transforms
from utils.loaders import get_loaders
from utils.losses import LabelSmoothingLoss

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

batch_size = 32

df = pd.read_csv("dataset_with_labels.csv")
df["weak_label"] = df.apply(lambda x: 0 if x.api_name == "no_logo" else 1 if x.n_images > 1 else -1, axis=1)
df = df[df.weak_label > -1]

weight = df.groupby("weak_label").count()["path"] / df.shape[0]
weight = torch.Tensor(weight.values).log()

train, test = train_test_split(df, test_size=0.2, stratify=df.weak_label, random_state=42)

train_transform, valid_transform = get_transforms(img_size=224)
train_loader, valid_loader = get_loaders(
    train,
    test,
    train_transform,
    valid_transform,
    batch_size=batch_size,
    num_workers=2,
    shuffle=True
)

#ce = nn.CrossEntropyLoss()
loss = LabelSmoothingLoss(num_classes=2, smoothing=0.2, weight=None)

model = EfficientNetModel(num_classes=2, loss=loss, weight=weight)
model_checkpoint = ModelCheckpoint(monitor="val_loss",
                                   verbose=True,
                                   dirpath="models/",
                                   mode="min",
                                   filename="{epoch}_{val_loss:.4f}")

trainer = pl.Trainer(gpus=1,
                     max_epochs=20,
                     auto_lr_find=True,
                     callbacks=[model_checkpoint])
trainer.fit(model, train_loader, valid_loader)

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

test["pred"] = pred
test.to_csv("test.csv", index=False)

precision = precision_score(test.weak_label, test.pred > 0.5)
recall = recall_score(test.weak_label, test.pred > 0.5)
f1 = f1_score(test.weak_label, test.pred > 0.5)
conf_matrix = confusion_matrix(test.weak_label, test.pred > 0.5)
print(f"Precision = {precision:.4f}")
print(f"Recall = {recall:.4f}")
print(f"F1 = {f1:.4f}")
print("Confusion matrix:")
print(conf_matrix)