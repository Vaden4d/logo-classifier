import os
import argparse
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint 

from utils.models import EfficientNetModel, EfficientNetSSL
from utils.transforms import get_transforms
from utils.loaders import get_loaders, get_loader
from utils.losses import LabelSmoothingLoss
from utils.misc import seed_everything, predict_on_loader
from utils.visualization import display_metrics
from utils.dataset import ImageDataset

from mixmatch_pytorch import MixMatchLoader, get_mixmatch_loss

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score

parser = argparse.ArgumentParser(description='PyTorch Lightning Training')
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='Number of total training epochs')
parser.add_argument('--batch_size', default=32, type=int, metavar='N',
                    help='Train and test batch size')   
parser.add_argument('--gpu', default=1, type=int,
                    help='0 if CPU mode, 1 if GPU')
parser.add_argument("--ssl", action="store_true",
                    help="Use semi-supervised pipeline")
parser.add_argument('--csv', default='dataset_with_weak_labels.csv', type=str,
                    help='Training .csv file with target column')
parser.add_argument('--target_column', default='weak_label', type=str,
                    help='Name of target column of the .csv file'
                    )
parser.add_argument('--validate', action="store_true",
                    help="Validate model on labeled dataset"
)
parser.add_argument('--validation_csv', default="labeled_part.csv",
                    help="Validation .csv file with labeled target"
)
parser.add_argument('--target_validation', default="label",
                    help="Name of target column in validation dataset"
)
parser.add_argument('--test_size', default=0.2, type=float,
                    help='Test dataset size'
                    )
parser.add_argument('--image_size', default=224, type=int,
                    help='Desired image size'
)
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of processes for PyTorch data loaders'
                    )
parser.add_argument('--random_state', default=42, type=int,
                    help='Random seed for all random operations'
                    )
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.gpu = args.gpu if torch.cuda.is_available() else 0
seed_everything(args.random_state)

# target_column has unique values in set -1, 0, 1
# -1 corresponds to the unlabeled data
df = pd.read_csv(args.csv)
labeled = df[df[args.target_column] > -1]
if args.ssl:
    print("Semi-supervised learning model is on...")
    unlabeled = df[df[args.target_column] == -1]

# weights to initialize bias of FC layer of classifier
weight = labeled.groupby(args.target_column).count()["path"] / labeled.shape[0]
weight = torch.Tensor(weight.values).log()

train_labeled, test_labeled = train_test_split(labeled, test_size=args.test_size, stratify=labeled[args.target_column], random_state=args.random_state)

train_transform, valid_transform = get_transforms(img_size=args.image_size)
train_labeled_loader, valid_labeled_loader = get_loaders(
    train_labeled,
    test_labeled,
    train_transform,
    valid_transform,
    target_column=args.target_column,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True
)

if args.ssl:
    dataset_unlabeled = ImageDataset(unlabeled, train_transform, target_column=None)

loss = LabelSmoothingLoss(num_classes=2, smoothing=0.2, weight=None)

if args.ssl:
    print("Semi-supervised learning model is configured...")
    model = EfficientNetSSL(loss=loss, num_classes=2, weight=weight)
else:
    model = EfficientNetModel(loss=loss, num_classes=2, weight=weight)

model_checkpoint = ModelCheckpoint(monitor="val_acc_f1",
                                   verbose=True,
                                   dirpath="models/",
                                   mode="max",
                                   filename="{epoch}_{val_acc_f1:.4f}")

if args.ssl:
    # SSL approach changes only train dataloader and model class
    train_loader = MixMatchLoader(
        train_labeled_loader,
        dataset_unlabeled,
        model,
        output_transform=nn.Softmax(dim=-1),
        K=2,
        T=0.5,
        alpha=0.75
    )
else:
    train_loader = train_labeled_loader

trainer = pl.Trainer(gpus=args.gpu,
                     max_epochs=args.epochs,
                     precision=16,
                     auto_lr_find=True,
                     callbacks=[model_checkpoint])
trainer.fit(model, train_loader, valid_labeled_loader)

test_labeled["pred"] = predict_on_loader(valid_labeled_loader, model, device)
test_labeled.to_csv("test_labeled_with_preds.csv", index=False)

# TODO
# threshold tuning
print("Metrics results on the test sample with weak labels:")
display_metrics(test_labeled[args.target_column], test_labeled["pred"], threshold=0.5)

if args.validate:

    validation = pd.read_csv(args.validation_csv)
    validation[args.target_validation] = validation[args.target_validation].apply(lambda x: 1 if x == "logo" else 0)

    labeled_loader = get_loader(
        validation,
        "label",
        valid_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False
    )

    validation["pred"] = predict_on_loader(labeled_loader, model, device)
    validation.to_csv("labeled_with_preds.csv", index=False)
    print("Metrics results on the labeled sample with strong labels:")
    display_metrics(validation["label"], validation["pred"], threshold=0.5)

