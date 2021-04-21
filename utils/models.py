import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from efficientnet_pytorch import EfficientNet
from pytorch_lightning.metrics.classification import F1
from mixmatch_pytorch import get_mixmatch_loss

class EfficientNetModel(pl.LightningModule):
    def __init__(self, loss, weight=None, num_classes=2):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b1', 
                                                      num_classes=num_classes)
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Linear(in_features, 
                                       num_classes)
        with torch.no_grad():
            if weight is not None:
                self.efficient_net._fc.bias.data = weight
        self.num_classes = num_classes
        self.loss = loss
        
        self.metric = F1(num_classes=self.num_classes, average="none")
        self.train_metric = self.metric.clone()
        self.val_metric = self.metric.clone()
    
    def forward(self,x):
        out = self.efficient_net(x)
        return out
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
  
    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["targets"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        f1_score = self.train_metric(y_hat, y.argmax(dim=-1))[1]
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_f1', self.train_metric.compute()[1], prog_bar=True, logger=True)
  
    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["targets"]
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        f1_score = self.val_metric(y_hat, y.argmax(dim=-1))[1]
        self.log("val_loss", loss, prog_bar=True, logger=True)

    def validation_epoch_end(self, outs):
        self.log('val_acc_f1', self.val_metric.compute()[1], prog_bar=True, logger=True)

class EfficientNetSSL(pl.LightningModule):
    def __init__(self, loss, weight=None, num_classes=2, T=0.5):
        super().__init__()
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b1', 
                                                      num_classes=num_classes)
        in_features = self.efficient_net._fc.in_features
        self.efficient_net._fc = nn.Linear(in_features, 
                                       num_classes)
        with torch.no_grad():
            if weight is not None:
                self.efficient_net._fc.bias.data = weight
        self.num_classes = num_classes
        self.criterion_labeled = loss
        self.loss = get_mixmatch_loss(
            criterion_labeled=self.criterion_labeled,
            output_transform=nn.Softmax(dim=-1),
            K=2,
            weight_unlabeled=1.0,
            criterion_unlabeled=nn.MSELoss()
        )

        self.train_metric = F1(num_classes=self.num_classes, average="none")
        self.val_metric = F1(num_classes=self.num_classes, average="none")
        
    def forward(self, x):
        out = self.efficient_net(x)
        return out
  
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
  
    def training_step(self, batch, batch_idx):
        x, y = batch["features"], batch["targets"]
        logits = self(x)
        loss = self.loss(logits, y)

        f1 = self.train_metric(logits, y.argmax(dim=-1))[1]

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log("train_f1", f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_f1', self.train_metric.compute()[1], prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch["features"], batch["targets"]
        logits = self(x)
        loss = self.criterion_labeled(logits, y)
        f1 = self.val_metric(logits, y.argmax(dim=-1))[1]

        self.log("val_loss", loss, prog_bar=True, logger=True)
        #self.log("val_f1", f1, prog_bar=True, logger=True)

    def validation_epoch_end(self, outs):
        self.log('val_acc_f1', self.val_metric.compute()[1], prog_bar=True, logger=True)