import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from .utils import *


class ResNet50MLC(nn.Module):
    """
    Pretrained Resnet-50 Image Multi-label Classification Model
    """

    def __init__(self, output_size):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3), nn.Linear(model.fc.in_features, output_size)
        )

        self.base_model = model
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.sigm(self.base_model(x))

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.binary_cross_entropy(out, labels)

        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.binary_cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.cpu().detach(), "val_acc": acc.cpu()}

    def validation_epoch_end(self, output):
        batch_losses = [x["val_loss"] for x in output]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x["val_acc"] for x in output]
        epoch_acc = torch.stack(batch_accs).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(
            "Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                result["lrs"][-1],
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )
